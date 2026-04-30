"""Plane-stress static FEM solver for a cantilever beam (shear-bending problem).

This script reproduces the example from section 2.1.2 of the textbook.
The same boundary-value problem is solved with six mesh cases (figure 2.3):

  Q4 element cases:
    Q1 - coarse regular grid          (nx=4, ny=2)
    Q2 - coarse distorted grid        (nx=4, ny=2, parallelogram-type distortion)
    Q3 - fine regular grid            (nx=8, ny=4)

  CST element cases:
    T1 - coarse grid, X-split         (nx=4, ny=2; each Q4 -> 4 CST via centroid node)
    T2 - coarse grid, diagonal split  (nx=4, ny=2; each Q4 -> 2 CST via single diagonal)
    T3 - fine grid, X-split           (nx=8, ny=4; each Q4 -> 4 CST via centroid node)

FEM results are compared against Euler-Bernoulli and Timoshenko beam theory.
For this problem L/h = 40/10 = 4, so the shear deformation correction from
Timoshenko theory is non-negligible.

Geometry (plane stress):
    Length  L = 40 mm, Height  h = 10 mm, Thickness b = 1 mm
    A = b * h = 10 mm^2,  I = b * h^3 / 12 = 1000/12 mm^4

Loading:
    Right face: uniform downward shear traction  tau = P/A = 1 N/mm^2
    Equivalent total shear force  P = tau * A = 10 N

Boundary conditions:
    Left face (x = 0): fully clamped  ux = uy = 0

Outputs are written to sec_02/2.1/2/outputs.
"""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

material_props: dict[str, Any] = {
    "E": 30000.0,  # MPa
    "nu": 0.30,
    "analysis_type": "plane_stress",
    "thickness": 1.0,  # mm (out-of-plane width b)
}

geometry: dict[str, Any] = {
    "length": 40.0,  # mm (beam span L, x-direction)
    "height": 10.0,  # mm (beam depth h, y-direction)
}

loads: dict[str, float] = {
    # Uniform shear traction on the right face (downward).
    # tau = P / A = 1 N/mm^2  =>  P = 10 N
    "right_traction_y": -1.0,
}

boundary_conditions: dict[str, Any] = {
    "left_edge_fixed": True,  # ux = uy = 0 on x = 0
}

# Six analysis cases matching figure 2.3 in the textbook.
analysis_cases: list[dict[str, str]] = [
    {"name": "Q1", "element_type": "q4", "mesh_type": "regular_coarse"},
    {"name": "Q2", "element_type": "q4", "mesh_type": "distorted_coarse"},
    {"name": "Q3", "element_type": "q4", "mesh_type": "regular_fine"},
    {"name": "T1", "element_type": "cst", "mesh_type": "regular_coarse", "tri_split": "x_split"},
    {"name": "T2", "element_type": "cst", "mesh_type": "regular_coarse", "tri_split": "diagonal"},
    {"name": "T3", "element_type": "cst", "mesh_type": "regular_fine", "tri_split": "x_split"},
]

mesh_resolutions: dict[str, dict[str, int]] = {
    "coarse": {"nx": 4, "ny": 2},
    "fine": {"nx": 8, "ny": 4},
}

# Parallelogram distortion applied to interior (non-boundary) nodes of the
# coarse grid when mesh_type == "distorted_coarse" (case Q2).
# Each node's x-coordinate is shifted by:
#   dx = distortion_factor * cell_width * (j / ny)
# where j is the node row index (0 at bottom, ny at top).
distortion_factor: float = 0.5

output_options: dict[str, Any] = {
    "output_dir": "sec_02/2.1/2/outputs",
    "displacement_scale": 500.0,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MeshData:
    """Structured mesh and boundary metadata for the beam."""

    nodes: np.ndarray  # shape (N, 2)  [x, y] coordinates
    q4_elements: np.ndarray  # shape (E, 4)  node indices per Q4 cell
    right_edges: np.ndarray  # shape (R, 2)  edge node pairs at x = L
    fixed_nodes: np.ndarray  # shape (F,)    node indices on x = 0


@dataclass
class AnalyticalSolution:
    """Beam theory reference values for the cantilever problem."""

    tip_deflection_EB_mm: float  # Euler-Bernoulli tip deflection
    tip_deflection_timoshenko_mm: float  # Timoshenko tip deflection (shear corrected)
    sigma_x_fixed_end_MPa: float  # max bending stress at x=0, y=+/-h/2
    tau_xy_neutral_MPa: float  # max shear stress at neutral axis (y=0)


@dataclass
class CaseResult:
    """Solver outputs for a single mesh case."""

    case_name: str
    element_type: str
    mesh_type: str
    solve_time_s: float
    displacements: np.ndarray
    reactions: np.ndarray
    external_force_vector: np.ndarray
    element_conn: np.ndarray
    element_stress: np.ndarray
    element_centers: np.ndarray
    von_mises: np.ndarray
    dof_count: int
    max_disp: float
    max_von_mises: float
    force_balance_error: float
    tip_deflection_mm: float
    sigma_x_fixed_end_MPa: float
    tau_xy_neutral_axis_MPa: float


# ---------------------------------------------------------------------------
# Constitutive matrix
# ---------------------------------------------------------------------------


def plane_stress_matrix(young: float, poisson: float) -> np.ndarray:
    """Build constitutive matrix for isotropic plane stress.

    For plane stress (thin plate, sigma_z = 0):
        D = E/(1-nu^2) * [[1,  nu, 0          ],
                          [nu, 1,  0          ],
                          [0,  0,  (1-nu)/2   ]]

    Compare with plane strain (sec_02/2.1/1) where sigma_z != 0 and the
    coefficient becomes E / ((1+nu)(1-2nu)).

    Args:
        young: Young's modulus in MPa.
        poisson: Poisson ratio.

    Returns:
        3x3 constitutive matrix in MPa units.
    """
    coef = young / (1.0 - poisson**2)
    return coef * np.array(
        [
            [1.0, poisson, 0.0],
            [poisson, 1.0, 0.0],
            [0.0, 0.0, 0.5 * (1.0 - poisson)],
        ],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# Analytical solution
# ---------------------------------------------------------------------------


def compute_analytical_solution(
    geom: dict[str, Any],
    mat: dict[str, Any],
    load: dict[str, float],
) -> AnalyticalSolution:
    """Compute beam theory reference values for the cantilever problem.

    Euler-Bernoulli tip deflection:
        delta_EB = P * L^3 / (3 * E * I)

    Timoshenko tip deflection (shear correction added):
        delta_T = P * L^3 / (3 * E * I)  +  P * L / (kappa * G * A)
        kappa = 5/6 for rectangular cross-section

    Bending stress at fixed end (x=0), extreme fibre (y = +/- h/2):
        sigma_x = M * y / I = P * L * (h/2) / I

    Shear stress at neutral axis (y = 0), parabolic distribution:
        tau_xy_max = P * h^2 / (8 * I)  (same along any cross-section)

    Args:
        geom: Geometry dict with 'length' and 'height' keys.
        mat: Material dict with 'E', 'nu', 'thickness' keys.
        load: Loads dict with 'right_traction_y' key.

    Returns:
        AnalyticalSolution with all reference values.
    """
    L = float(geom["length"])
    h = float(geom["height"])
    b = float(mat["thickness"])
    E = float(mat["E"])
    nu = float(mat["nu"])

    A = b * h
    I = b * h**3 / 12.0
    G = E / (2.0 * (1.0 + nu))
    kappa = 5.0 / 6.0  # shear correction factor for rectangular section

    # Total shear force P (downward, positive value).
    # right_traction_y is negative (downward) => P = |tau| * A
    P = abs(float(load["right_traction_y"])) * A

    delta_EB = P * L**3 / (3.0 * E * I)
    delta_timoshenko = delta_EB + P * L / (kappa * G * A)

    sigma_x_max = P * L * (h / 2.0) / I
    tau_xy_max = P * h**2 / (8.0 * I)

    LOGGER.info(
        "Analytical solution: delta_EB=%.6f mm, delta_T=%.6f mm, sigma_x=%.4f MPa, tau_xy=%.4f MPa",
        delta_EB,
        delta_timoshenko,
        sigma_x_max,
        tau_xy_max,
    )

    return AnalyticalSolution(
        tip_deflection_EB_mm=delta_EB,
        tip_deflection_timoshenko_mm=delta_timoshenko,
        sigma_x_fixed_end_MPa=sigma_x_max,
        tau_xy_neutral_MPa=tau_xy_max,
    )


# ---------------------------------------------------------------------------
# Mesh generation
# ---------------------------------------------------------------------------


def generate_beam_mesh(case_cfg: dict[str, str]) -> MeshData:
    """Generate a rectangular beam mesh for the given case configuration.

    Supports three mesh variants:
      - regular_coarse / regular_fine : uniform rectangular grid
      - distorted_coarse              : parallelogram distortion (case Q2)

    For distorted_coarse, interior nodes (not on left/right boundary) have
    their x-coordinates shifted by:
        dx = distortion_factor * cell_width * (j / ny)
    where j is the node row index.  This produces the parallelogram pattern
    shown in figure 2.3 [Q2].

    Args:
        case_cfg: Dict with 'mesh_type' key.

    Returns:
        MeshData with nodes, q4_elements, right_edges, fixed_nodes.
    """
    mesh_type = case_cfg["mesh_type"]
    resolution_key = "fine" if "fine" in mesh_type else "coarse"
    nx = int(mesh_resolutions[resolution_key]["nx"])
    ny = int(mesh_resolutions[resolution_key]["ny"])

    L = float(geometry["length"])
    h = float(geometry["height"])

    x_coords = np.linspace(0.0, L, nx + 1)
    y_coords = np.linspace(0.0, h, ny + 1)
    xx, yy = np.meshgrid(x_coords, y_coords, indexing="xy")
    nodes = np.column_stack((xx.ravel(), yy.ravel()))

    def node_id(i: int, j: int) -> int:
        """Global node index for column i, row j (both 0-based)."""
        return j * (nx + 1) + i

    # Apply parallelogram distortion for Q2.
    if mesh_type == "distorted_coarse":
        cell_w = L / nx
        for j in range(ny + 1):
            for i in range(nx + 1):
                # Do not shift nodes on the left (i=0) or right (i=nx) face
                # so that the boundary geometry remains exact.
                if i == 0 or i == nx:
                    continue
                nid = node_id(i, j)
                shift = distortion_factor * cell_w * (j / ny)
                nodes[nid, 0] += shift

    # Build Q4 connectivity (counter-clockwise: SW, SE, NE, NW).
    q4_elements: list[list[int]] = []
    for j in range(ny):
        for i in range(nx):
            n_sw = node_id(i, j)
            n_se = node_id(i + 1, j)
            n_ne = node_id(i + 1, j + 1)
            n_nw = node_id(i, j + 1)
            q4_elements.append([n_sw, n_se, n_ne, n_nw])

    # Right-face edges for traction application (x = L).
    right_edges: list[list[int]] = []
    for j in range(ny):
        right_edges.append([node_id(nx, j), node_id(nx, j + 1)])

    # Fixed nodes on left face (x = 0).
    fixed_nodes = np.array([node_id(0, j) for j in range(ny + 1)], dtype=int)

    return MeshData(
        nodes=nodes,
        q4_elements=np.asarray(q4_elements, dtype=int),
        right_edges=np.asarray(right_edges, dtype=int),
        fixed_nodes=fixed_nodes,
    )


# ---------------------------------------------------------------------------
# CST element functions
# ---------------------------------------------------------------------------


def cst_b_matrix(coords: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute CST strain-displacement matrix and element area.

    Args:
        coords: Shape (3, 2) array of node (x, y) coordinates.

    Returns:
        Tuple of (B matrix (3x6), area).
    """
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]

    two_area = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    area = 0.5 * abs(two_area)
    if area <= 0.0:
        raise ValueError("Non-positive CST element area.")

    b1 = y2 - y3
    b2 = y3 - y1
    b3 = y1 - y2
    c1 = x3 - x2
    c2 = x1 - x3
    c3 = x2 - x1

    b_mat = (1.0 / (2.0 * area)) * np.array(
        [
            [b1, 0.0, b2, 0.0, b3, 0.0],
            [0.0, c1, 0.0, c2, 0.0, c3],
            [c1, b1, c2, b2, c3, b3],
        ],
        dtype=float,
    )
    return b_mat, area


def cst_stiffness(coords: np.ndarray, constitutive: np.ndarray, thickness: float) -> np.ndarray:
    """Compute 6x6 CST element stiffness matrix.

    Args:
        coords: Shape (3, 2) node coordinates.
        constitutive: 3x3 constitutive matrix.
        thickness: Out-of-plane thickness b.

    Returns:
        6x6 stiffness matrix.
    """
    b_mat, area = cst_b_matrix(coords)
    return thickness * area * (b_mat.T @ constitutive @ b_mat)


# ---------------------------------------------------------------------------
# Q4 element functions
# ---------------------------------------------------------------------------


def q4_shape_derivatives(xi: float, eta: float) -> np.ndarray:
    """Return shape function derivatives dN/d(xi,eta) for Q4 in parent coords.

    Node ordering: SW(-1,-1), SE(+1,-1), NE(+1,+1), NW(-1,+1).

    Returns:
        Shape (4, 2) array; row i is [dNi/dxi, dNi/deta].
    """
    return 0.25 * np.array(
        [
            [-(1.0 - eta), -(1.0 - xi)],
            [+(1.0 - eta), -(1.0 + xi)],
            [+(1.0 + eta), +(1.0 + xi)],
            [-(1.0 + eta), +(1.0 - xi)],
        ],
        dtype=float,
    )


def q4_b_matrix(coords: np.ndarray, xi: float, eta: float) -> tuple[np.ndarray, float]:
    """Build Q4 strain-displacement matrix and Jacobian determinant.

    Args:
        coords: Shape (4, 2) node coordinates.
        xi, eta: Parent-space coordinates.

    Returns:
        Tuple of (B matrix (3x8), det(J)).
    """
    dnd_parent = q4_shape_derivatives(xi, eta)
    jacobian = coords.T @ dnd_parent
    det_j = float(np.linalg.det(jacobian))
    if det_j <= 0.0:
        raise ValueError(f"Invalid Q4 Jacobian (det={det_j:.6e}). Check mesh distortion.")

    inv_j = np.linalg.inv(jacobian)
    dnd_global = dnd_parent @ inv_j

    b_mat = np.zeros((3, 8), dtype=float)
    for i in range(4):
        dnx = dnd_global[i, 0]
        dny = dnd_global[i, 1]
        base = 2 * i
        b_mat[0, base] = dnx
        b_mat[1, base + 1] = dny
        b_mat[2, base] = dny
        b_mat[2, base + 1] = dnx

    return b_mat, det_j


def q4_stiffness(coords: np.ndarray, constitutive: np.ndarray, thickness: float) -> np.ndarray:
    """Compute 8x8 Q4 element stiffness matrix using 2x2 Gauss integration.

    Args:
        coords: Shape (4, 2) node coordinates.
        constitutive: 3x3 constitutive matrix.
        thickness: Out-of-plane thickness b.

    Returns:
        8x8 stiffness matrix.
    """
    gauss = 1.0 / np.sqrt(3.0)
    gauss_points = [(-gauss, -gauss), (gauss, -gauss), (gauss, gauss), (-gauss, gauss)]
    ke = np.zeros((8, 8), dtype=float)
    for xi, eta in gauss_points:
        b_mat, det_j = q4_b_matrix(coords, xi, eta)
        ke += thickness * (b_mat.T @ constitutive @ b_mat) * det_j
    return ke


# ---------------------------------------------------------------------------
# CST splitting utilities
# ---------------------------------------------------------------------------


def q4_to_2cst(q4_elements: np.ndarray) -> np.ndarray:
    """Split each Q4 into 2 CST triangles using a single diagonal (SW-NE).

    This is the same pattern used in sec_02/2.1/1 (T2 case).

    Args:
        q4_elements: Shape (E, 4) with columns [SW, SE, NE, NW].

    Returns:
        Shape (2E, 3) triangle connectivity.
    """
    tri_list: list[list[int]] = []
    for n_sw, n_se, n_ne, n_nw in q4_elements:
        tri_list.append([n_sw, n_se, n_ne])
        tri_list.append([n_sw, n_ne, n_nw])
    return np.asarray(tri_list, dtype=int)


def q4_to_4cst(
    q4_elements: np.ndarray,
    nodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split each Q4 into 4 CST triangles by inserting a centroid node (X-split).

    A centroid node is added at the geometric centre of each Q4 cell.
    Four triangles are formed: (SW,SE,C), (SE,NE,C), (NE,NW,C), (NW,SW,C).
    This produces the [T1] / [T3] patterns from figure 2.3.

    Args:
        q4_elements: Shape (E, 4) with columns [SW, SE, NE, NW].
        nodes: Existing node coordinates, shape (N, 2).

    Returns:
        Tuple of:
          - new_nodes (N + E, 2): original nodes extended with centroid nodes
          - triangles (4E, 3): triangle connectivity referencing new_nodes
    """
    new_nodes = list(nodes)
    tri_list: list[list[int]] = []

    for conn in q4_elements:
        n_sw, n_se, n_ne, n_nw = conn
        centroid = np.mean(nodes[conn], axis=0)
        c_idx = len(new_nodes)
        new_nodes.append(centroid)

        tri_list.append([n_sw, n_se, c_idx])
        tri_list.append([n_se, n_ne, c_idx])
        tri_list.append([n_ne, n_nw, c_idx])
        tri_list.append([n_nw, n_sw, c_idx])

    return np.asarray(new_nodes, dtype=float), np.asarray(tri_list, dtype=int)


# ---------------------------------------------------------------------------
# System assembly
# ---------------------------------------------------------------------------


def assemble_system(
    mesh: MeshData,
    element_type: str,
    tri_split: str,
    constitutive: np.ndarray,
    thickness: float,
    traction_y: float,
) -> tuple[object, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble global stiffness matrix, force vector and element connectivity.

    For CST cases, the Q4 elements are first split into triangles using either
    the diagonal or X-split pattern.  For the X-split, new centroid nodes are
    appended to the node array; this extended array is returned as the second
    element of the tuple.

    Args:
        mesh: MeshData with nodes and Q4 connectivity.
        element_type: 'q4' or 'cst'.
        tri_split: 'diagonal' or 'x_split' (used only when element_type='cst').
        constitutive: 3x3 constitutive matrix.
        thickness: Out-of-plane thickness.
        traction_y: Right-face shear traction in N/mm^2 (negative = downward).

    Returns:
        Tuple of (stiffness CSR matrix, force vector, element connectivity,
                  nodes array possibly extended with centroid nodes).
    """
    nodes = mesh.nodes.copy()

    if element_type == "cst":
        if tri_split == "x_split":
            nodes, elements = q4_to_4cst(mesh.q4_elements, nodes)
        else:
            elements = q4_to_2cst(mesh.q4_elements)
        dof_per_element = 6
    else:
        elements = mesh.q4_elements
        dof_per_element = 8

    ndof = 2 * nodes.shape[0]
    stiffness = lil_matrix((ndof, ndof), dtype=float)
    force = np.zeros(ndof, dtype=float)

    for conn in elements:
        coords = nodes[conn]
        if element_type == "cst":
            ke = cst_stiffness(coords, constitutive, thickness)
        else:
            ke = q4_stiffness(coords, constitutive, thickness)

        dofs = np.zeros(dof_per_element, dtype=int)
        for local_idx, nid in enumerate(conn):
            dofs[2 * local_idx] = 2 * nid
            dofs[2 * local_idx + 1] = 2 * nid + 1

        for i_loc, i_glo in enumerate(dofs):
            for j_loc, j_glo in enumerate(dofs):
                stiffness[i_glo, j_glo] += ke[i_loc, j_loc]

    # Apply right-face shear traction as equivalent nodal forces.
    # Right-edge nodes are in the original mesh (centroid nodes are interior).
    for n1, n2 in mesh.right_edges:
        x1, y1 = mesh.nodes[n1]
        x2, y2 = mesh.nodes[n2]
        edge_length = float(np.hypot(x2 - x1, y2 - y1))
        nodal_fy = traction_y * thickness * edge_length / 2.0
        force[2 * n1 + 1] += nodal_fy
        force[2 * n2 + 1] += nodal_fy

    return stiffness.tocsr(), force, elements, nodes


def apply_boundary_conditions(
    fixed_nodes: np.ndarray,
    total_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build fixed and free DOF arrays for a fully clamped left edge.

    Args:
        fixed_nodes: Node indices on the clamped edge (ux = uy = 0).
        total_nodes: Total number of nodes (including centroid nodes if any).

    Returns:
        Tuple of (fixed DOF indices, free DOF indices).
    """
    fixed_dofs: list[int] = []
    for nid in fixed_nodes:
        fixed_dofs.append(2 * int(nid))
        fixed_dofs.append(2 * int(nid) + 1)

    fixed = np.unique(np.asarray(fixed_dofs, dtype=int))
    all_dofs = np.arange(2 * total_nodes, dtype=int)
    free = np.setdiff1d(all_dofs, fixed)
    return fixed, free


# ---------------------------------------------------------------------------
# Stress computation
# ---------------------------------------------------------------------------


def von_mises_plane_stress(sigma_x: float, sigma_y: float, tau_xy: float) -> float:
    """Compute von Mises stress for plane-stress state (sigma_z = 0).

    Args:
        sigma_x, sigma_y: Normal stresses in MPa.
        tau_xy: Shear stress in MPa.

    Returns:
        Von Mises equivalent stress in MPa.
    """
    return float(np.sqrt(sigma_x**2 - sigma_x * sigma_y + sigma_y**2 + 3.0 * tau_xy**2))


def compute_element_stress(
    nodes: np.ndarray,
    elements: np.ndarray,
    displacements: np.ndarray,
    constitutive: np.ndarray,
    element_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute element-centre stress and von Mises values.

    For CST elements the strain (and hence stress) is constant within each
    element, so the centre value equals the element value everywhere.

    For Q4 elements the stress is averaged over the four Gauss points.

    Args:
        nodes: Shape (N, 2) node coordinates.
        elements: Shape (E, n_nodes_per_elem) connectivity.
        displacements: Flattened displacement vector (2N,).
        constitutive: 3x3 constitutive matrix.
        element_type: 'cst' or 'q4'.

    Returns:
        Tuple of (stress (E,3), centers (E,2), von_mises (E,)).
    """
    elem_stress: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    vm_list: list[float] = []

    for conn in elements:
        coords = nodes[conn]
        ue = np.zeros(2 * conn.size, dtype=float)
        for loc_i, nid in enumerate(conn):
            ue[2 * loc_i] = displacements[2 * nid]
            ue[2 * loc_i + 1] = displacements[2 * nid + 1]

        if element_type == "cst":
            b_mat, _ = cst_b_matrix(coords)
            sigma = constitutive @ (b_mat @ ue)
        else:
            gauss = 1.0 / np.sqrt(3.0)
            gp = [(-gauss, -gauss), (gauss, -gauss), (gauss, gauss), (-gauss, gauss)]
            sigmas = []
            for xi, eta in gp:
                b_mat, _ = q4_b_matrix(coords, xi, eta)
                sigmas.append(constitutive @ (b_mat @ ue))
            sigma = np.mean(np.asarray(sigmas), axis=0)

        sxx, syy, txy = float(sigma[0]), float(sigma[1]), float(sigma[2])
        vm = von_mises_plane_stress(sxx, syy, txy)

        elem_stress.append(np.array([sxx, syy, txy], dtype=float))
        centers.append(np.mean(coords, axis=0))
        vm_list.append(vm)

    return np.asarray(elem_stress), np.asarray(centers), np.asarray(vm_list)


# ---------------------------------------------------------------------------
# Cross-section extraction
# ---------------------------------------------------------------------------


def extract_cross_section_stress(
    element_centers: np.ndarray,
    element_stress: np.ndarray,
    x_target: float,
    x_tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract element stresses near a given x-coordinate cross-section.

    Args:
        element_centers: Shape (E, 2) element centroid coordinates.
        element_stress: Shape (E, 3) [sigma_x, sigma_y, tau_xy].
        x_target: Target x-coordinate in mm.
        x_tol: Half-width of the extraction band in mm.

    Returns:
        Tuple of (y_coords (M,), sigma_x (M,), tau_xy (M,)) sorted by y.
    """
    mask = np.abs(element_centers[:, 0] - x_target) <= x_tol
    if not np.any(mask):
        return np.array([]), np.array([]), np.array([])

    y_vals = element_centers[mask, 1]
    sigma_x = element_stress[mask, 0]
    tau_xy = element_stress[mask, 2]

    order = np.argsort(y_vals)
    return y_vals[order], sigma_x[order], tau_xy[order]


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


def solve_case(
    case_cfg: dict[str, str],
    constitutive: np.ndarray,
    analytical: AnalyticalSolution,
) -> tuple[CaseResult, np.ndarray]:
    """Solve one FEM case and compute all metrics.

    Returns:
        Tuple of (CaseResult, extended nodes array).
        The extended nodes array may contain additional centroid nodes for
        X-split CST cases.
    """
    case_name = case_cfg["name"]
    element_type = case_cfg["element_type"]
    mesh_type = case_cfg["mesh_type"]
    tri_split = case_cfg.get("tri_split", "diagonal")

    thickness = float(material_props["thickness"])
    traction_y = float(loads["right_traction_y"])

    mesh = generate_beam_mesh(case_cfg)

    start = time.perf_counter()
    stiffness, force, elements, nodes = assemble_system(mesh, element_type, tri_split, constitutive, thickness, traction_y)
    fixed, free = apply_boundary_conditions(mesh.fixed_nodes, nodes.shape[0])

    displacements = np.zeros(2 * nodes.shape[0], dtype=float)
    kff = stiffness[free][:, free]
    ff = force[free]
    displacements[free] = spsolve(kff, ff)

    reactions = stiffness @ displacements - force
    elapsed = time.perf_counter() - start

    stress, centers, von_mises = compute_element_stress(nodes, elements, displacements, constitutive, element_type)

    # Tip deflection: mean uy of all right-face nodes in original mesh.
    right_node_ids = np.unique(mesh.right_edges.ravel())
    tip_deflections = [displacements[2 * nid + 1] for nid in right_node_ids]
    # Take the magnitude of the mean (traction is downward -> uy is negative)
    tip_deflection = float(abs(np.mean(tip_deflections)))

    # Cross-section stress at fixed end (x ≈ 0).
    L = float(geometry["length"])
    h = float(geometry["height"])
    cell_w = L / mesh_resolutions["coarse" if "coarse" in mesh_type else "fine"]["nx"]
    x_tol = cell_w * 0.6

    y_vals, sxx_vals, _ = extract_cross_section_stress(centers, stress, x_target=cell_w * 0.5, x_tol=x_tol)
    if sxx_vals.size > 0:
        sigma_x_fixed = float(np.max(np.abs(sxx_vals)))
    else:
        sigma_x_fixed = 0.0

    # Shear stress at neutral axis (y ≈ h/2) at mid-span.
    x_mid = L / 2.0
    y_mid_section, _, tau_vals = extract_cross_section_stress(centers, stress, x_target=x_mid, x_tol=x_tol)
    if tau_vals.size > 0:
        neutral_mask = np.abs(y_mid_section - h / 2.0) <= (h / 2.0 * 0.5)
        if np.any(neutral_mask):
            tau_neutral = float(np.mean(np.abs(tau_vals[neutral_mask])))
        else:
            tau_neutral = float(np.max(np.abs(tau_vals)))
    else:
        tau_neutral = 0.0

    nodal_u = displacements.reshape(-1, 2)
    max_disp = float(np.max(np.linalg.norm(nodal_u, axis=1)))

    # Force balance: sum of reactions + sum of external forces = 0
    external_fy = float(np.sum(force[1::2]))
    fixed_y_dofs = np.array([2 * int(nid) + 1 for nid in mesh.fixed_nodes], dtype=int)
    reaction_fy = float(np.sum(reactions[fixed_y_dofs]))
    balance_error = abs(external_fy + reaction_fy) / max(abs(external_fy), 1.0)

    LOGGER.info(
        "Case %s: tip_u=%.6f mm (EB=%.6f, T=%.6f), sigma_x=%.3f MPa, tau_xy=%.3f MPa, vm_max=%.3f MPa, balance=%.2e, t=%.3fs",
        case_name,
        tip_deflection,
        analytical.tip_deflection_EB_mm,
        analytical.tip_deflection_timoshenko_mm,
        sigma_x_fixed,
        tau_neutral,
        float(np.max(von_mises)),
        balance_error,
        elapsed,
    )

    result = CaseResult(
        case_name=case_name,
        element_type=element_type,
        mesh_type=mesh_type,
        solve_time_s=elapsed,
        displacements=displacements,
        reactions=reactions,
        external_force_vector=force,
        element_conn=elements,
        element_stress=stress,
        element_centers=centers,
        von_mises=von_mises,
        dof_count=displacements.size,
        max_disp=max_disp,
        max_von_mises=float(np.max(von_mises)),
        force_balance_error=float(balance_error),
        tip_deflection_mm=tip_deflection,
        sigma_x_fixed_end_MPa=sigma_x_fixed,
        tau_xy_neutral_axis_MPa=tau_neutral,
    )
    return result, nodes


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_case(
    mesh: MeshData,
    nodes: np.ndarray,
    result: CaseResult,
    output_dir: Path,
) -> None:
    """Save displacement and von Mises stress plots for a single case.

    Args:
        mesh: Original mesh data (for boundary reference).
        nodes: Node array used in this case (may include centroid nodes).
        result: Solved case result.
        output_dir: Directory to save figures.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Case {result.case_name}", fontsize=11)

    # Displacement magnitude
    disp_magnitude = np.linalg.norm(result.displacements.reshape(-1, 2), axis=1)
    ax0 = axes[0]
    ax0.set_title("Displacement magnitude [mm]")
    ax0.set_aspect("equal")

    # Triangulate for pcolor
    triang = Triangulation(nodes[:, 0], nodes[:, 1])
    disp_interp = disp_magnitude[: nodes.shape[0]]
    tc0 = ax0.tripcolor(triang, disp_interp, shading="gouraud", cmap="viridis")
    plt.colorbar(tc0, ax=ax0)
    ax0.set_xlabel("x [mm]")
    ax0.set_ylabel("y [mm]")

    # Von Mises stress (element-centred)
    ax1 = axes[1]
    ax1.set_title("Von Mises stress [MPa]")
    ax1.set_aspect("equal")

    cx = result.element_centers[:, 0]
    cy = result.element_centers[:, 1]
    vm = result.von_mises

    triang2 = Triangulation(cx, cy)
    tc1 = ax1.tripcolor(triang2, vm, shading="flat", cmap="hot_r")
    plt.colorbar(tc1, ax=ax1)
    ax1.set_xlabel("x [mm]")

    fig.tight_layout()
    fname = output_dir / f"case_{result.case_name}_fields.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    LOGGER.info("Saved %s", fname)


def plot_stress_profiles(
    results: list[CaseResult],
    analytical: AnalyticalSolution,
    output_dir: Path,
) -> None:
    """Plot sigma_x distribution at x=0 and tau_xy distribution at x=L/2.

    Args:
        results: All solved cases.
        analytical: Analytical solution.
        output_dir: Save directory.
    """
    h = float(geometry["height"])
    L = float(geometry["length"])
    b = float(material_props["thickness"])
    P = abs(float(loads["right_traction_y"]) * h * b)
    I = b * h**3 / 12.0

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Cross-section stress profiles", fontsize=11)

    # Theoretical sigma_x at x=0: sigma_x = M*z/I, M = P*L, z = y - h/2
    y_th = np.linspace(0.0, h, 100)
    z_th = y_th - h / 2.0
    sxx_th = P * L * z_th / I
    ax0.plot(sxx_th, y_th, "k--", linewidth=2, label="Beam theory")
    ax0.set_title("sigma_x at x ≈ 0 (fixed end)")
    ax0.set_xlabel("sigma_x [MPa]")
    ax0.set_ylabel("y [mm]")

    # Theoretical tau_xy at x=L/2: parabolic
    tau_th = 1.5 * P / (b * h) * (1.0 - (2.0 * z_th / h) ** 2)
    ax1.plot(tau_th, y_th, "k--", linewidth=2, label="Beam theory")
    ax1.set_title("tau_xy at x = L/2")
    ax1.set_xlabel("tau_xy [MPa]")
    ax1.set_ylabel("y [mm]")

    for res in results:
        coarse = "coarse" in res.mesh_type
        nx = mesh_resolutions["coarse" if coarse else "fine"]["nx"]
        cell_w = L / nx
        x_tol = cell_w * 0.6

        y0, sxx0, _ = extract_cross_section_stress(res.element_centers, res.element_stress, x_target=cell_w * 0.5, x_tol=x_tol)
        if y0.size:
            ax0.plot(sxx0, y0, marker=".", markersize=4, label=res.case_name, alpha=0.8)

        y1, _, tau1 = extract_cross_section_stress(res.element_centers, res.element_stress, x_target=L / 2.0, x_tol=x_tol)
        if y1.size:
            ax1.plot(tau1, y1, marker=".", markersize=4, label=res.case_name, alpha=0.8)

    ax0.legend(fontsize=7)
    ax1.legend(fontsize=7)
    fig.tight_layout()
    fname = output_dir / "stress_profiles.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    LOGGER.info("Saved %s", fname)


def plot_comparison_summary(
    results: list[CaseResult],
    analytical: AnalyticalSolution,
    output_dir: Path,
) -> None:
    """Save bar-chart comparison of tip deflection and peak stresses.

    Args:
        results: All solved cases.
        analytical: Analytical (beam theory) reference values.
        output_dir: Save directory.
    """
    names = [r.case_name for r in results]
    tip_fem = [r.tip_deflection_mm for r in results]
    sxx_fem = [r.sigma_x_fixed_end_MPa for r in results]
    tau_fem = [r.tau_xy_neutral_axis_MPa for r in results]
    x = np.arange(len(names))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("FEM vs Analytical comparison", fontsize=11)

    # Tip deflection
    ax = axes[0]
    ax.bar(x, tip_fem, color="steelblue", label="FEM")
    ax.axhline(analytical.tip_deflection_EB_mm, color="green", linestyle="--", linewidth=1.5, label="EB theory")
    ax.axhline(analytical.tip_deflection_timoshenko_mm, color="orange", linestyle="-.", linewidth=1.5, label="Timoshenko")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, fontsize=8)
    ax.set_title("Tip deflection [mm]")
    ax.legend(fontsize=8)

    # sigma_x at fixed end
    ax = axes[1]
    ax.bar(x, sxx_fem, color="firebrick", label="FEM")
    ax.axhline(abs(analytical.sigma_x_fixed_end_MPa), color="k", linestyle="--", linewidth=1.5, label="Beam theory")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, fontsize=8)
    ax.set_title("|sigma_x| at fixed end [MPa]")
    ax.legend(fontsize=8)

    # tau_xy neutral axis
    ax = axes[2]
    ax.bar(x, tau_fem, color="darkorange", label="FEM")
    ax.axhline(analytical.tau_xy_neutral_MPa, color="k", linestyle="--", linewidth=1.5, label="Beam theory")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, fontsize=8)
    ax.set_title("tau_xy neutral axis [MPa]")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fname = output_dir / "comparison_summary.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    LOGGER.info("Saved %s", fname)


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def save_comparison(
    results: list[CaseResult],
    analytical: AnalyticalSolution,
    output_dir: Path,
) -> None:
    """Write comparison_metrics.csv.

    Args:
        results: All solved cases.
        analytical: Analytical beam theory values.
        output_dir: Directory to write the CSV file.
    """
    csv_path = output_dir / "comparison_metrics.csv"
    header = (
        "case,element_type,mesh_type,dof,"
        "tip_deflection_mm,ref_EB_mm,ref_timoshenko_mm,error_vs_timoshenko_pct,"
        "sigma_x_fixed_end_MPa,ref_sigma_x_MPa,"
        "tau_xy_neutral_MPa,ref_tau_xy_MPa,"
        "max_von_mises_MPa,force_balance_error,solve_time_s"
    )
    lines = [header]
    for r in results:
        ref_t = analytical.tip_deflection_timoshenko_mm
        err_t = abs(r.tip_deflection_mm - ref_t) / ref_t * 100.0 if ref_t else 0.0
        line = (
            f"{r.case_name},{r.element_type},{r.mesh_type},{r.dof_count},"
            f"{r.tip_deflection_mm:.6f},{analytical.tip_deflection_EB_mm:.6f},"
            f"{analytical.tip_deflection_timoshenko_mm:.6f},{err_t:.3f},"
            f"{r.sigma_x_fixed_end_MPa:.4f},{abs(analytical.sigma_x_fixed_end_MPa):.4f},"
            f"{r.tau_xy_neutral_axis_MPa:.4f},{analytical.tau_xy_neutral_MPa:.4f},"
            f"{r.max_von_mises:.4f},{r.force_balance_error:.4e},{r.solve_time_s:.4f}"
        )
        lines.append(line)

    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Saved %s", csv_path)


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------


def run() -> None:
    """Execute all six FEM cases, save plots, CSV and log summary table."""
    output_dir = Path(output_options["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Output directory ready: %s", output_dir)

    # Analytical solution
    analytical = compute_analytical_solution(geometry, material_props, loads)
    LOGGER.info(
        "Analytical: EB=%.6f mm, Timoshenko=%.6f mm, sigma_x_max=%.4f MPa, tau_max=%.4f MPa",
        analytical.tip_deflection_EB_mm,
        analytical.tip_deflection_timoshenko_mm,
        analytical.sigma_x_fixed_end_MPa,
        analytical.tau_xy_neutral_MPa,
    )

    constitutive = plane_stress_matrix(
        float(material_props["E"]),
        float(material_props["nu"]),
    )

    results: list[CaseResult] = []
    nodes_per_case: list[np.ndarray] = []
    meshes: list[MeshData] = []

    for case_cfg in analysis_cases:
        case_name = case_cfg["name"]
        LOGGER.info("=== Solving case: %s ===", case_name)
        mesh = generate_beam_mesh(case_cfg)
        meshes.append(mesh)
        result, nodes_used = solve_case(case_cfg, constitutive, analytical)
        results.append(result)
        nodes_per_case.append(nodes_used)

    # Per-case field plots
    if output_options.get("plot_case_fields", True):
        for mesh, result, nodes_used in zip(meshes, results, nodes_per_case):
            plot_case(mesh, nodes_used, result, output_dir)

    # Cross-section stress profiles
    plot_stress_profiles(results, analytical, output_dir)

    # Bar chart comparison
    plot_comparison_summary(results, analytical, output_dir)

    # CSV summary
    save_comparison(results, analytical, output_dir)

    # Console summary table
    header_fmt = f"{'Case':<12}{'EType':<6}{'Mesh':<18}{'DOF':>6}  {'Tip[mm]':>10}  {'Err_T%':>8}  {'SxxMax':>8}  {'TauN':>8}  {'Balance':>10}  {'t[s]':>7}"
    LOGGER.info("Summary:\n%s", header_fmt)
    for r in results:
        ref_t = analytical.tip_deflection_timoshenko_mm
        err = abs(r.tip_deflection_mm - ref_t) / ref_t * 100.0 if ref_t else 0.0
        row = (
            f"{r.case_name:<12}{r.element_type:<6}{r.mesh_type:<18}{r.dof_count:>6}  "
            f"{r.tip_deflection_mm:>10.6f}  {err:>8.3f}  "
            f"{r.sigma_x_fixed_end_MPa:>8.4f}  {r.tau_xy_neutral_axis_MPa:>8.4f}  "
            f"{r.force_balance_error:>10.2e}  {r.solve_time_s:>7.4f}"
        )
        LOGGER.info(row)


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    try:
        run()
    except Exception as exc:
        LOGGER.exception("FEM execution failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
