"""Plane-stress static FEM solver for a cantilever beam (shear-bending problem).

This script reproduces the example from section 2.1.2 of the textbook.

The implementation is being migrated to the textbook's original separation:

    - mesh pattern (figure 2.3): Q1, Q2, Q3, T1, T2, T3
    - element version (figure 2.2): QL, QH, TL, TH, ...

This file currently implements the linear and serendipity quadrilateral versions:

    - Q1_QL, Q2_QL, Q3_QL   (4-node bilinear quadrilateral)
    - Q1_QH, Q2_QH, Q3_QH   (8-node serendipity quadrilateral)
    - T1_TL, T2_TL, T3_TL   (3-node constant-strain triangle)

The important correction is that T1/T2/T3 are now generated as direct
triangle meshes following figure 2.3, instead of being produced by splitting
an intermediate quadrilateral mesh afterwards.

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

# Current implemented cases: figure 2.3 mesh patterns with linear, QH, and TH versions.
analysis_cases: list[dict[str, str]] = [
    {"name": "Q1_QL", "mesh_pattern": "Q1", "element_version": "QL"},
    {"name": "Q2_QL", "mesh_pattern": "Q2", "element_version": "QL"},
    {"name": "Q3_QL", "mesh_pattern": "Q3", "element_version": "QL"},
    {"name": "Q1_QH", "mesh_pattern": "Q1", "element_version": "QH"},
    {"name": "Q2_QH", "mesh_pattern": "Q2", "element_version": "QH"},
    {"name": "Q3_QH", "mesh_pattern": "Q3", "element_version": "QH"},
    {"name": "T1_TL", "mesh_pattern": "T1", "element_version": "TL"},
    {"name": "T2_TL", "mesh_pattern": "T2", "element_version": "TL"},
    {"name": "T3_TL", "mesh_pattern": "T3", "element_version": "TL"},
    {"name": "T1_TH", "mesh_pattern": "T1", "element_version": "TH"},
    {"name": "T2_TH", "mesh_pattern": "T2", "element_version": "TH"},
    {"name": "T3_TH", "mesh_pattern": "T3", "element_version": "TH"},
]

# Pattern-specific grid counts following figure 2.3.
mesh_pattern_cfg: dict[str, dict[str, int]] = {
    "Q1": {"nx": 4, "ny": 1},
    "Q2": {"nx": 4, "ny": 1},
    "Q3": {"nx": 16, "ny": 4},
    "T1": {"nx": 4, "ny": 1},
    "T2": {"nx": 4, "ny": 1},
    "T3": {"nx": 16, "ny": 4},
}

# Parallelogram distortion applied to figure 2.3 distorted patterns Q2/T2.
distortion_factor_q2: float = 0.40
distortion_factor_t2_top: float = 0.18
distortion_factor_t2_bottom: float = 0.32

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
    quad_elements: np.ndarray | None  # shape (E, 4)  node indices per Q4 cell
    tri_elements: np.ndarray | None  # shape (E, 3)  node indices per triangle
    right_edges: np.ndarray  # shape (R, 2)  edge node pairs at x = L
    fixed_nodes: np.ndarray  # shape (F,)    node indices on x = 0
    mesh_pattern: str
    characteristic_dx: float
    q8_elements: np.ndarray | None = None  # shape (E, 8)  node indices per Q8 cell
    t6_elements: np.ndarray | None = None  # shape (E, 6)  node indices per T6 cell
    right_edge_midnodes: np.ndarray | None = None  # shape (R,)  midside node per right edge


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
    mesh_pattern: str
    element_version: str
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
    nodal_vm: np.ndarray | None = None  # Von Mises averaged at corner nodes (for smooth plot)


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


def _build_rectangular_nodes(nx: int, ny: int) -> np.ndarray:
    """Build structured node coordinates on the beam rectangle."""
    L = float(geometry["length"])
    h = float(geometry["height"])
    x_coords = np.linspace(0.0, L, nx + 1)
    y_coords = np.linspace(0.0, h, ny + 1)
    xx, yy = np.meshgrid(x_coords, y_coords, indexing="xy")
    return np.column_stack((xx.ravel(), yy.ravel()))


def _structured_node_id(nx: int, i: int, j: int) -> int:
    """Return global node id for a structured grid."""
    return j * (nx + 1) + i


def _build_quad_mesh(pattern: str) -> MeshData:
    """Build direct quadrilateral mesh patterns Q1/Q2/Q3 from figure 2.3."""
    cfg = mesh_pattern_cfg[pattern]
    nx = int(cfg["nx"])
    ny = int(cfg["ny"])
    L = float(geometry["length"])

    nodes = _build_rectangular_nodes(nx, ny)

    if pattern == "Q2":
        cell_w = L / nx
        for i in range(1, nx):
            bottom_id = _structured_node_id(nx, i, 0)
            nodes[bottom_id, 0] += distortion_factor_q2 * cell_w

    quad_elements: list[list[int]] = []
    for j in range(ny):
        for i in range(nx):
            quad_elements.append(
                [
                    _structured_node_id(nx, i, j),
                    _structured_node_id(nx, i + 1, j),
                    _structured_node_id(nx, i + 1, j + 1),
                    _structured_node_id(nx, i, j + 1),
                ]
            )

    right_edges = np.asarray(
        [[_structured_node_id(nx, nx, j), _structured_node_id(nx, nx, j + 1)] for j in range(ny)],
        dtype=int,
    )
    fixed_nodes = np.asarray([_structured_node_id(nx, 0, j) for j in range(ny + 1)], dtype=int)

    return MeshData(
        nodes=nodes,
        quad_elements=np.asarray(quad_elements, dtype=int),
        tri_elements=None,
        right_edges=right_edges,
        fixed_nodes=fixed_nodes,
        mesh_pattern=pattern,
        characteristic_dx=L / nx,
    )


def _build_triangle_mesh(pattern: str) -> MeshData:
    """Build direct triangle mesh patterns T1/T2/T3 from figure 2.3."""
    cfg = mesh_pattern_cfg[pattern]
    nx = int(cfg["nx"])
    ny = int(cfg["ny"])
    L = float(geometry["length"])
    h = float(geometry["height"])

    nodes = _build_rectangular_nodes(nx, ny)
    cell_w = L / nx

    if pattern == "T2":
        for i in range(1, nx):
            bottom_id = _structured_node_id(nx, i, 0)
            top_id = _structured_node_id(nx, i, ny)
            nodes[bottom_id, 0] += distortion_factor_t2_bottom * cell_w
            nodes[top_id, 0] -= distortion_factor_t2_top * cell_w

    tri_elements: list[list[int]] = []
    for j in range(ny):
        for i in range(nx):
            n_sw = _structured_node_id(nx, i, j)
            n_se = _structured_node_id(nx, i + 1, j)
            n_ne = _structured_node_id(nx, i + 1, j + 1)
            n_nw = _structured_node_id(nx, i, j + 1)

            use_rising_diagonal = (i + j) % 2 == 0
            if use_rising_diagonal:
                tri_elements.append([n_sw, n_se, n_ne])
                tri_elements.append([n_sw, n_ne, n_nw])
            else:
                tri_elements.append([n_sw, n_se, n_nw])
                tri_elements.append([n_se, n_ne, n_nw])

    right_edges = np.asarray(
        [[_structured_node_id(nx, nx, j), _structured_node_id(nx, nx, j + 1)] for j in range(ny)],
        dtype=int,
    )
    fixed_nodes = np.asarray([_structured_node_id(nx, 0, j) for j in range(ny + 1)], dtype=int)

    return MeshData(
        nodes=nodes,
        quad_elements=None,
        tri_elements=np.asarray(tri_elements, dtype=int),
        right_edges=right_edges,
        fixed_nodes=fixed_nodes,
        mesh_pattern=pattern,
        characteristic_dx=cell_w,
    )


def generate_beam_mesh(case_cfg: dict[str, str]) -> MeshData:
    """Generate one textbook mesh pattern directly from figure 2.3.

    For QH the base Q4 mesh is enriched with midside nodes to produce Q8.
    For TH the base TL mesh is enriched with midside nodes to produce T6.
    """
    mesh_pattern = case_cfg["mesh_pattern"]
    element_version = case_cfg["element_version"]
    if mesh_pattern.startswith("Q"):
        base_mesh = _build_quad_mesh(mesh_pattern)
        if element_version == "QH":
            return enrich_q4_to_q8(base_mesh)
        return base_mesh
    base_mesh = _build_triangle_mesh(mesh_pattern)
    if element_version == "TH":
        return enrich_tri3_to_tri6(base_mesh)
    return base_mesh


# ---------------------------------------------------------------------------
# Q8 mesh enrichment
# ---------------------------------------------------------------------------


def enrich_q4_to_q8(mesh: MeshData) -> MeshData:
    """Add midside nodes to a Q4 mesh to produce a Q8 (serendipity) mesh.

    Each edge shared between two Q4 elements gets exactly one midside node.
    Boundary edges on the right face and fixed left face are handled
    consistently so traction and constraint DOFs are correct.

    Node ordering in each Q8 element:
        [n_sw, n_se, n_ne, n_nw, m_s, m_e, m_n, m_w]

    Args:
        mesh: Q4 MeshData (must have quad_elements set).

    Returns:
        New MeshData with extended nodes, q8_elements, right_edge_midnodes,
        and fixed_nodes augmented by left-edge midside nodes.

    Raises:
        ValueError: If mesh has no quadrilateral connectivity.
    """
    if mesh.quad_elements is None:
        raise ValueError("enrich_q4_to_q8 requires quad_elements to be set.")

    new_nodes: list[np.ndarray] = list(mesh.nodes)
    edge_to_midnode: dict[tuple[int, int], int] = {}

    def get_midnode(n1: int, n2: int) -> int:
        """Return (creating if needed) the midside node index for edge (n1,n2)."""
        key = (min(n1, n2), max(n1, n2))
        if key not in edge_to_midnode:
            mid = (mesh.nodes[n1] + mesh.nodes[n2]) * 0.5
            idx = len(new_nodes)
            new_nodes.append(mid)
            edge_to_midnode[key] = idx
        return edge_to_midnode[key]

    # Build Q8 connectivity.  Q4 ordering: [SW, SE, NE, NW]
    q8_list: list[list[int]] = []
    for conn in mesh.quad_elements:
        n_sw, n_se, n_ne, n_nw = int(conn[0]), int(conn[1]), int(conn[2]), int(conn[3])
        m_s = get_midnode(n_sw, n_se)  # South midside
        m_e = get_midnode(n_se, n_ne)  # East  midside
        m_n = get_midnode(n_ne, n_nw)  # North midside
        m_w = get_midnode(n_nw, n_sw)  # West  midside
        q8_list.append([n_sw, n_se, n_ne, n_nw, m_s, m_e, m_n, m_w])

    # Right-edge midside nodes (one per original right edge pair).
    right_mid: list[int] = []
    for n1, n2 in mesh.right_edges:
        right_mid.append(get_midnode(int(n1), int(n2)))

    # Fixed (left-edge) midside nodes.
    sorted_fixed = np.sort(mesh.fixed_nodes)
    left_mid: list[int] = []
    for i in range(len(sorted_fixed) - 1):
        left_mid.append(get_midnode(int(sorted_fixed[i]), int(sorted_fixed[i + 1])))

    new_fixed = np.concatenate([mesh.fixed_nodes, np.asarray(left_mid, dtype=int)])

    return MeshData(
        nodes=np.asarray(new_nodes, dtype=float),
        quad_elements=mesh.quad_elements,
        tri_elements=None,
        q8_elements=np.asarray(q8_list, dtype=int),
        right_edges=mesh.right_edges,
        right_edge_midnodes=np.asarray(right_mid, dtype=int),
        fixed_nodes=new_fixed,
        mesh_pattern=mesh.mesh_pattern,
        characteristic_dx=mesh.characteristic_dx,
    )


# ---------------------------------------------------------------------------
# T6 mesh enrichment
# ---------------------------------------------------------------------------


def enrich_tri3_to_tri6(mesh: MeshData) -> MeshData:
    """Add midside nodes to a TL mesh to produce a T6 (quadratic triangle) mesh.

    Each edge shared between two triangles gets exactly one midside node.
    Right-face and left-face boundary edges are handled so traction and
    constraint DOFs are consistent.

    Node ordering in each T6 element:
        [n1, n2, n3, m12, m23, m31]
    where m12 is the midpoint of edge n1-n2, etc.

    Args:
        mesh: TL MeshData (must have tri_elements set).

    Returns:
        New MeshData with extended nodes, t6_elements, right_edge_midnodes,
        and fixed_nodes augmented by left-edge midside nodes.

    Raises:
        ValueError: If mesh has no triangle connectivity.
    """
    if mesh.tri_elements is None:
        raise ValueError("enrich_tri3_to_tri6 requires tri_elements to be set.")

    new_nodes: list[np.ndarray] = list(mesh.nodes)
    edge_to_midnode: dict[tuple[int, int], int] = {}

    def get_midnode(n1: int, n2: int) -> int:
        """Return (creating if needed) the midside node index for edge (n1,n2)."""
        key = (min(n1, n2), max(n1, n2))
        if key not in edge_to_midnode:
            mid = (mesh.nodes[n1] + mesh.nodes[n2]) * 0.5
            idx = len(new_nodes)
            new_nodes.append(mid)
            edge_to_midnode[key] = idx
        return edge_to_midnode[key]

    # Build T6 connectivity.  TL ordering: [n1, n2, n3]
    t6_list: list[list[int]] = []
    for conn in mesh.tri_elements:
        n1, n2, n3 = int(conn[0]), int(conn[1]), int(conn[2])
        m12 = get_midnode(n1, n2)
        m23 = get_midnode(n2, n3)
        m31 = get_midnode(n3, n1)
        t6_list.append([n1, n2, n3, m12, m23, m31])

    # Right-edge midside nodes (one per original right edge pair).
    right_mid: list[int] = []
    for n1, n2 in mesh.right_edges:
        right_mid.append(get_midnode(int(n1), int(n2)))

    # Fixed (left-edge) midside nodes.
    sorted_fixed = np.sort(mesh.fixed_nodes)
    left_mid: list[int] = []
    for i in range(len(sorted_fixed) - 1):
        left_mid.append(get_midnode(int(sorted_fixed[i]), int(sorted_fixed[i + 1])))

    new_fixed = np.concatenate([mesh.fixed_nodes, np.asarray(left_mid, dtype=int)])

    return MeshData(
        nodes=np.asarray(new_nodes, dtype=float),
        quad_elements=None,
        tri_elements=mesh.tri_elements,
        t6_elements=np.asarray(t6_list, dtype=int),
        right_edges=mesh.right_edges,
        right_edge_midnodes=np.asarray(right_mid, dtype=int),
        fixed_nodes=new_fixed,
        mesh_pattern=mesh.mesh_pattern,
        characteristic_dx=mesh.characteristic_dx,
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
# Q8 serendipity element functions
# ---------------------------------------------------------------------------


def q8_shape_derivatives(xi: float, eta: float) -> np.ndarray:
    """Return shape function derivatives dN/d(xi,eta) for Q8 serendipity.

    Node ordering:
        0 = SW (-1,-1), 1 = SE (+1,-1), 2 = NE (+1,+1), 3 = NW (-1,+1),
        4 = S  ( 0,-1), 5 = E  (+1, 0), 6 = N  ( 0,+1), 7 = W  (-1, 0).

    Shape functions:
        Corner i: N_i = (1/4)(1 + xi_i*xi)(1 + eta_i*eta)(xi_i*xi + eta_i*eta - 1)
        Mid-S (4): N_4 = (1/2)(1 - xi^2)(1 - eta)
        Mid-E (5): N_5 = (1/2)(1 + xi)(1 - eta^2)
        Mid-N (6): N_6 = (1/2)(1 - xi^2)(1 + eta)
        Mid-W (7): N_7 = (1/2)(1 - xi)(1 - eta^2)

    Args:
        xi: Parent-space xi coordinate.
        eta: Parent-space eta coordinate.

    Returns:
        Shape (8, 2) array; row i is [dNi/dxi, dNi/deta].
    """
    dN = np.zeros((8, 2), dtype=float)

    # Corner node derivatives: N_i = (1/4)(1+xi_i*xi)(1+eta_i*eta)(xi_i*xi+eta_i*eta-1)
    # dN_0/dxi  = (1/4)(1-eta)(2xi+eta),  dN_0/deta = (1/4)(1-xi)(xi+2eta)
    dN[0, 0] = 0.25 * (1.0 - eta) * (2.0 * xi + eta)
    dN[0, 1] = 0.25 * (1.0 - xi) * (xi + 2.0 * eta)

    # dN_1/dxi  = (1/4)(1-eta)(2xi-eta),  dN_1/deta = (1/4)(1+xi)(-xi+2eta)
    dN[1, 0] = 0.25 * (1.0 - eta) * (2.0 * xi - eta)
    dN[1, 1] = 0.25 * (1.0 + xi) * (-xi + 2.0 * eta)

    # dN_2/dxi  = (1/4)(1+eta)(2xi+eta),  dN_2/deta = (1/4)(1+xi)(xi+2eta)
    dN[2, 0] = 0.25 * (1.0 + eta) * (2.0 * xi + eta)
    dN[2, 1] = 0.25 * (1.0 + xi) * (xi + 2.0 * eta)

    # dN_3/dxi  = (1/4)(1+eta)(2xi-eta),  dN_3/deta = (1/4)(1-xi)(-xi+2eta)
    dN[3, 0] = 0.25 * (1.0 + eta) * (2.0 * xi - eta)
    dN[3, 1] = 0.25 * (1.0 - xi) * (-xi + 2.0 * eta)

    # Midside node derivatives
    # N_4 = (1/2)(1-xi^2)(1-eta):  dN_4/dxi = -xi(1-eta),  dN_4/deta = -(1-xi^2)/2
    dN[4, 0] = -xi * (1.0 - eta)
    dN[4, 1] = -0.5 * (1.0 - xi**2)

    # N_5 = (1/2)(1+xi)(1-eta^2):  dN_5/dxi = (1-eta^2)/2,  dN_5/deta = -(1+xi)*eta
    dN[5, 0] = 0.5 * (1.0 - eta**2)
    dN[5, 1] = -(1.0 + xi) * eta

    # N_6 = (1/2)(1-xi^2)(1+eta):  dN_6/dxi = -xi(1+eta),  dN_6/deta = (1-xi^2)/2
    dN[6, 0] = -xi * (1.0 + eta)
    dN[6, 1] = 0.5 * (1.0 - xi**2)

    # N_7 = (1/2)(1-xi)(1-eta^2):  dN_7/dxi = -(1-eta^2)/2,  dN_7/deta = -(1-xi)*eta
    dN[7, 0] = -0.5 * (1.0 - eta**2)
    dN[7, 1] = -(1.0 - xi) * eta

    return dN


def q8_b_matrix(coords: np.ndarray, xi: float, eta: float) -> tuple[np.ndarray, float]:
    """Build Q8 strain-displacement matrix and Jacobian determinant.

    Args:
        coords: Shape (8, 2) node coordinates in the order defined by
            q8_shape_derivatives (corners SW,SE,NE,NW then midsides S,E,N,W).
        xi, eta: Parent-space coordinates.

    Returns:
        Tuple of (B matrix (3x16), det(J)).
    """
    dnd_parent = q8_shape_derivatives(xi, eta)  # (8, 2)
    jacobian = coords.T @ dnd_parent  # (2, 2)
    det_j = float(np.linalg.det(jacobian))
    if det_j <= 0.0:
        raise ValueError(f"Invalid Q8 Jacobian (det={det_j:.6e}). Check mesh distortion.")

    inv_j = np.linalg.inv(jacobian)
    dnd_global = dnd_parent @ inv_j  # (8, 2)

    b_mat = np.zeros((3, 16), dtype=float)
    for i in range(8):
        dnx = dnd_global[i, 0]
        dny = dnd_global[i, 1]
        base = 2 * i
        b_mat[0, base] = dnx
        b_mat[1, base + 1] = dny
        b_mat[2, base] = dny
        b_mat[2, base + 1] = dnx

    return b_mat, det_j


def q8_stiffness(coords: np.ndarray, constitutive: np.ndarray, thickness: float) -> np.ndarray:
    """Compute 16x16 Q8 element stiffness matrix using 3x3 Gauss integration.

    Three-point Gauss quadrature is sufficient for exact integration of the
    rational integrands arising from non-rectangular (distorted) Q8 elements.

    Args:
        coords: Shape (8, 2) node coordinates.
        constitutive: 3x3 constitutive matrix.
        thickness: Out-of-plane thickness b.

    Returns:
        16x16 stiffness matrix.
    """
    gp = np.sqrt(3.0 / 5.0)
    w1, w2 = 5.0 / 9.0, 8.0 / 9.0
    gauss_points = [
        (-gp, -gp, w1 * w1),
        (-gp, 0.0, w1 * w2),
        (-gp, gp, w1 * w1),
        (0.0, -gp, w2 * w1),
        (0.0, 0.0, w2 * w2),
        (0.0, gp, w2 * w1),
        (+gp, -gp, w1 * w1),
        (+gp, 0.0, w1 * w2),
        (+gp, gp, w1 * w1),
    ]
    ke = np.zeros((16, 16), dtype=float)
    for xi, eta, w in gauss_points:
        b_mat, det_j = q8_b_matrix(coords, xi, eta)
        ke += thickness * w * (b_mat.T @ constitutive @ b_mat) * det_j
    return ke


# ---------------------------------------------------------------------------
# T6 quadratic triangle element functions
# ---------------------------------------------------------------------------


def t6_shape_derivatives(L1: float, L2: float) -> np.ndarray:
    """Return shape function derivatives dN/d(L1,L2) for T6 in area coordinates.

    Node ordering (matches enrich_tri3_to_tri6):
        0: n1 (corner, L1=1, L2=0, L3=0)
        1: n2 (corner, L1=0, L2=1, L3=0)
        2: n3 (corner, L1=0, L2=0, L3=1)
        3: m12 (midside n1-n2, L1=L2=0.5)
        4: m23 (midside n2-n3, L2=L3=0.5)
        5: m31 (midside n3-n1, L3=L1=0.5)

    With L3 = 1 - L1 - L2, independent variables are L1 and L2.

    Shape functions:
        N0 = L1(2L1 - 1),  N1 = L2(2L2 - 1),  N2 = L3(2L3 - 1)
        N3 = 4 L1 L2,      N4 = 4 L2 L3,       N5 = 4 L1 L3

    Args:
        L1: First area coordinate.
        L2: Second area coordinate.

    Returns:
        Shape (6, 2) array; row i is [dNi/dL1, dNi/dL2].
    """
    L3 = 1.0 - L1 - L2
    dN = np.zeros((6, 2), dtype=float)

    # Corner node 0: N0 = L1(2L1 - 1)
    dN[0, 0] = 4.0 * L1 - 1.0
    dN[0, 1] = 0.0

    # Corner node 1: N1 = L2(2L2 - 1)
    dN[1, 0] = 0.0
    dN[1, 1] = 4.0 * L2 - 1.0

    # Corner node 2: N2 = L3(2L3 - 1), L3 = 1 - L1 - L2
    #   dN2/dL1 = (4L3 - 1) * (-1) = 1 - 4L3
    #   dN2/dL2 = (4L3 - 1) * (-1) = 1 - 4L3
    dN[2, 0] = 1.0 - 4.0 * L3
    dN[2, 1] = 1.0 - 4.0 * L3

    # Midside node 3: N3 = 4 L1 L2
    dN[3, 0] = 4.0 * L2
    dN[3, 1] = 4.0 * L1

    # Midside node 4: N4 = 4 L2 L3 = 4 L2 (1 - L1 - L2)
    dN[4, 0] = -4.0 * L2
    dN[4, 1] = 4.0 * (L3 - L2)

    # Midside node 5: N5 = 4 L1 L3 = 4 L1 (1 - L1 - L2)
    dN[5, 0] = 4.0 * (L3 - L1)
    dN[5, 1] = -4.0 * L1

    return dN


def t6_b_matrix(coords: np.ndarray, L1: float, L2: float) -> tuple[np.ndarray, float]:
    """Build T6 strain-displacement matrix and Jacobian determinant.

    The Jacobian maps from area coordinates (L1, L2) to physical (x, y):
        J = dN^T @ coords   (shape 2x2)

    Args:
        coords: Shape (6, 2) node coordinates in T6 ordering.
        L1, L2: Area coordinates; L3 = 1 - L1 - L2.

    Returns:
        Tuple of (B matrix (3x12), det(J)).
    """
    dnd_natural = t6_shape_derivatives(L1, L2)  # (6, 2)
    # Same convention as q4_b_matrix: jacobian = coords.T @ dN/d(natural) = J^T.
    # Then inv_j = (J^T)^{-1} = J^{-T}, and dnd_global = dnd_natural @ inv_j
    # correctly yields dNi/dx, dNi/dy via the chain rule.
    jacobian = coords.T @ dnd_natural  # (2, 2) = J^T
    det_j = float(np.linalg.det(jacobian))
    if det_j <= 0.0:
        raise ValueError(f"Invalid T6 Jacobian (det={det_j:.6e}). Check mesh distortion.")

    inv_j = np.linalg.inv(jacobian)
    dnd_global = dnd_natural @ inv_j  # (6, 2)

    b_mat = np.zeros((3, 12), dtype=float)
    for i in range(6):
        dnx = dnd_global[i, 0]
        dny = dnd_global[i, 1]
        base = 2 * i
        b_mat[0, base] = dnx
        b_mat[1, base + 1] = dny
        b_mat[2, base] = dny
        b_mat[2, base + 1] = dnx

    return b_mat, det_j


def t6_stiffness(coords: np.ndarray, constitutive: np.ndarray, thickness: float) -> np.ndarray:
    """Compute 12x12 T6 element stiffness matrix using 3-point Gauss integration.

    Uses the standard 3-point rule on the reference triangle (degree 2 exactness):
        points (L1, L2): (1/6, 1/6), (2/3, 1/6), (1/6, 2/3)
        weights: 1/6 each  (sum = 1/2 = area of reference triangle)

    K_e = t * sum_gp [ w * B^T D B * det(J) ]

    Args:
        coords: Shape (6, 2) node coordinates.
        constitutive: 3x3 constitutive matrix.
        thickness: Out-of-plane thickness b.

    Returns:
        12x12 stiffness matrix.
    """
    a = 1.0 / 6.0
    b = 2.0 / 3.0
    gauss_points = [(a, a, a), (b, a, a), (a, b, a)]  # (L1, L2, weight)
    ke = np.zeros((12, 12), dtype=float)
    for L1, L2, w in gauss_points:
        b_mat, det_j = t6_b_matrix(coords, L1, L2)
        ke += thickness * w * (b_mat.T @ constitutive @ b_mat) * det_j
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
    element_version: str,
    constitutive: np.ndarray,
    thickness: float,
    traction_y: float,
) -> tuple[object, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble global stiffness matrix and force vector for one case."""
    nodes = mesh.nodes

    if element_version == "TL":
        if mesh.tri_elements is None:
            raise ValueError(f"Triangle connectivity missing for pattern {mesh.mesh_pattern}.")
        elements = mesh.tri_elements
        dof_per_element = 6
    elif element_version == "QL":
        if mesh.quad_elements is None:
            raise ValueError(f"Quadrilateral connectivity missing for pattern {mesh.mesh_pattern}.")
        elements = mesh.quad_elements
        dof_per_element = 8
    elif element_version == "QH":
        if mesh.q8_elements is None:
            raise ValueError(f"Q8 connectivity missing for pattern {mesh.mesh_pattern}.")
        elements = mesh.q8_elements
        dof_per_element = 16
    elif element_version == "TH":
        if mesh.t6_elements is None:
            raise ValueError(f"T6 connectivity missing for pattern {mesh.mesh_pattern}.")
        elements = mesh.t6_elements
        dof_per_element = 12
    else:
        raise NotImplementedError(f"Element version {element_version} is not implemented yet.")

    ndof = 2 * nodes.shape[0]
    stiffness = lil_matrix((ndof, ndof), dtype=float)
    force = np.zeros(ndof, dtype=float)

    for conn in elements:
        coords = nodes[conn]
        if element_version == "TL":
            ke = cst_stiffness(coords, constitutive, thickness)
        elif element_version == "QL":
            ke = q4_stiffness(coords, constitutive, thickness)
        elif element_version == "QH":
            ke = q8_stiffness(coords, constitutive, thickness)
        else:
            ke = t6_stiffness(coords, constitutive, thickness)

        dofs = np.zeros(dof_per_element, dtype=int)
        for local_idx, nid in enumerate(conn):
            dofs[2 * local_idx] = 2 * nid
            dofs[2 * local_idx + 1] = 2 * nid + 1

        for i_loc, i_glo in enumerate(dofs):
            for j_loc, j_glo in enumerate(dofs):
                stiffness[i_glo, j_glo] += ke[i_loc, j_loc]

    # Apply right-face shear traction as equivalent nodal forces.
    # For QL/TL: uniform traction on a linear edge -> each corner node gets L/2.
    # For QH/TH: quadratic edge with 3 nodes -> consistent nodal forces [1/6, 2/3, 1/6]*L.
    use_quadratic_edge = element_version in ("QH", "TH")
    for edge_idx, (n1, n2) in enumerate(mesh.right_edges):
        x1, y1 = mesh.nodes[n1]
        x2, y2 = mesh.nodes[n2]
        edge_length = float(np.hypot(x2 - x1, y2 - y1))
        if use_quadratic_edge and mesh.right_edge_midnodes is not None:
            # Consistent nodal force for 3-node quadratic edge under uniform traction.
            n_mid = int(mesh.right_edge_midnodes[edge_idx])
            force[2 * n1 + 1] += traction_y * thickness * edge_length / 6.0
            force[2 * n2 + 1] += traction_y * thickness * edge_length / 6.0
            force[2 * n_mid + 1] += traction_y * thickness * edge_length * (2.0 / 3.0)
        else:
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
    element_version: str,
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
        element_version: 'TL', 'QL', 'QH', or 'TH'.

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

        if element_version == "TL":
            b_mat, _ = cst_b_matrix(coords)
            sigma = constitutive @ (b_mat @ ue)
        elif element_version == "QL":
            gauss = 1.0 / np.sqrt(3.0)
            gp = [(-gauss, -gauss), (gauss, -gauss), (gauss, gauss), (-gauss, gauss)]
            sigmas = []
            for xi, eta in gp:
                b_mat, _ = q4_b_matrix(coords, xi, eta)
                sigmas.append(constitutive @ (b_mat @ ue))
            sigma = np.mean(np.asarray(sigmas), axis=0)
        elif element_version == "QH":
            # Average stress over 2x2 Gauss points (sufficient for stress output).
            g = 1.0 / np.sqrt(3.0)
            gp_qh = [(-g, -g), (g, -g), (g, g), (-g, g)]
            sigmas = []
            for xi, eta in gp_qh:
                b_mat, _ = q8_b_matrix(coords, xi, eta)
                sigmas.append(constitutive @ (b_mat @ ue))
            sigma = np.mean(np.asarray(sigmas), axis=0)
        else:
            # TH: evaluate at element centroid (L1 = L2 = L3 = 1/3).
            b_mat, _ = t6_b_matrix(coords, 1.0 / 3.0, 1.0 / 3.0)
            sigma = constitutive @ (b_mat @ ue)

        sxx, syy, txy = float(sigma[0]), float(sigma[1]), float(sigma[2])
        vm = von_mises_plane_stress(sxx, syy, txy)

        elem_stress.append(np.array([sxx, syy, txy], dtype=float))
        centers.append(np.mean(coords, axis=0))
        vm_list.append(vm)

    return np.asarray(elem_stress), np.asarray(centers), np.asarray(vm_list)


def compute_nodal_vm(
    nodes: np.ndarray,
    elements: np.ndarray,
    displacements: np.ndarray,
    constitutive: np.ndarray,
    element_version: str,
) -> np.ndarray:
    """Compute Von Mises stress averaged at each corner node.

    For each element the stress is evaluated at its corner nodes in natural
    coordinates, then accumulated and averaged over all elements sharing a
    node.  This produces a smooth nodal field suitable for Gouraud shading,
    which correctly shows bending stress varying through the element height
    even when ny=1 (a known limitation of centroid-based evaluation).

    Args:
        nodes: Shape (N, 2) node coordinates.
        elements: Shape (E, n_nodes_per_elem) connectivity.
        displacements: Flattened displacement vector (2N,).
        constitutive: 3x3 constitutive matrix.
        element_version: 'TL', 'QL', 'QH', or 'TH'.

    Returns:
        nodal_vm: Shape (N,) Von Mises stress at each node (0 for unreferenced).
    """
    # Natural coordinates of corner nodes for each element type.
    # Q4/Q8: (xi, eta) at SW, SE, NE, NW corners.
    _q4_corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
    # T3/T6: (L1, L2) barycentric at vertex 1, 2, 3 (L3 = 1-L1-L2).
    _t_corners = [(1.0, 0.0), (0.0, 1.0), (0.0, 0.0)]

    n_nodes = nodes.shape[0]
    vm_sum = np.zeros(n_nodes, dtype=float)
    vm_cnt = np.zeros(n_nodes, dtype=int)

    for conn in elements:
        coords = nodes[conn]
        ue = np.zeros(2 * conn.size, dtype=float)
        for loc_i, nid in enumerate(conn):
            ue[2 * loc_i] = displacements[2 * nid]
            ue[2 * loc_i + 1] = displacements[2 * nid + 1]

        if element_version == "TL":
            # CST: constant strain/stress - same value at all 3 corners.
            b_mat, _ = cst_b_matrix(coords)
            sigma = constitutive @ (b_mat @ ue)
            vm = von_mises_plane_stress(float(sigma[0]), float(sigma[1]), float(sigma[2]))
            for nid in conn:
                vm_sum[nid] += vm
                vm_cnt[nid] += 1

        elif element_version == "QL":
            for (xi, eta), nid in zip(_q4_corners, conn):
                b_mat, _ = q4_b_matrix(coords, xi, eta)
                sigma = constitutive @ (b_mat @ ue)
                vm = von_mises_plane_stress(float(sigma[0]), float(sigma[1]), float(sigma[2]))
                vm_sum[nid] += vm
                vm_cnt[nid] += 1

        elif element_version == "QH":
            # Corner nodes are the first 4 entries; midside nodes are conn[4:8].
            for (xi, eta), nid in zip(_q4_corners, conn[:4]):
                b_mat, _ = q8_b_matrix(coords, xi, eta)
                sigma = constitutive @ (b_mat @ ue)
                vm = von_mises_plane_stress(float(sigma[0]), float(sigma[1]), float(sigma[2]))
                vm_sum[nid] += vm
                vm_cnt[nid] += 1

        else:  # TH
            # Corner nodes are the first 3 entries; midside nodes are conn[3:6].
            for (L1, L2), nid in zip(_t_corners, conn[:3]):
                b_mat, _ = t6_b_matrix(coords, L1, L2)
                sigma = constitutive @ (b_mat @ ue)
                vm = von_mises_plane_stress(float(sigma[0]), float(sigma[1]), float(sigma[2]))
                vm_sum[nid] += vm
                vm_cnt[nid] += 1

    mask = vm_cnt > 0
    nodal_vm = np.zeros(n_nodes, dtype=float)
    nodal_vm[mask] = vm_sum[mask] / vm_cnt[mask]
    return nodal_vm


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
    mesh_pattern = case_cfg["mesh_pattern"]
    element_version = case_cfg["element_version"]

    thickness = float(material_props["thickness"])
    traction_y = float(loads["right_traction_y"])

    mesh = generate_beam_mesh(case_cfg)

    start = time.perf_counter()
    stiffness, force, elements, nodes = assemble_system(mesh, element_version, constitutive, thickness, traction_y)
    fixed, free = apply_boundary_conditions(mesh.fixed_nodes, nodes.shape[0])

    displacements = np.zeros(2 * nodes.shape[0], dtype=float)
    kff = stiffness[free][:, free]
    ff = force[free]
    displacements[free] = spsolve(kff, ff)

    reactions = stiffness @ displacements - force
    elapsed = time.perf_counter() - start

    stress, centers, von_mises = compute_element_stress(nodes, elements, displacements, constitutive, element_version)
    nodal_vm = compute_nodal_vm(nodes, elements, displacements, constitutive, element_version)

    # Tip deflection: mean uy of all right-face nodes in original mesh.
    right_node_ids = np.unique(mesh.right_edges.ravel())
    tip_deflections = [displacements[2 * nid + 1] for nid in right_node_ids]
    # Take the magnitude of the mean (traction is downward -> uy is negative)
    tip_deflection = float(abs(np.mean(tip_deflections)))

    # Cross-section stress at fixed end (x ≈ 0).
    L = float(geometry["length"])
    h = float(geometry["height"])
    cell_w = mesh.characteristic_dx
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
        mesh_pattern=mesh_pattern,
        element_version=element_version,
        solve_time_s=elapsed,
        displacements=displacements,
        reactions=reactions,
        external_force_vector=force,
        element_conn=elements,
        element_stress=stress,
        element_centers=centers,
        von_mises=von_mises,
        nodal_vm=nodal_vm,
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


def build_plot_triangles(mesh: MeshData) -> np.ndarray:
    """Build triangle connectivity for plotting from the explicit mesh topology.

    For Q8 meshes the plot uses the 4 corner nodes of each element (same
    decomposition as Q4) so that the mesh outline is visualised correctly
    without adding centroid nodes just for display purposes.
    """
    if mesh.tri_elements is not None:
        return mesh.tri_elements

    # Use Q4 corners, or the first 4 nodes of each Q8 element (corner nodes).
    quad_conn = mesh.quad_elements
    if quad_conn is None and mesh.q8_elements is not None:
        quad_conn = mesh.q8_elements[:, :4]

    if quad_conn is None:
        raise ValueError(f"Mesh pattern {mesh.mesh_pattern} has no plottable connectivity.")

    triangles: list[list[int]] = []
    for row in quad_conn:
        n_sw, n_se, n_ne, n_nw = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        triangles.append([n_sw, n_se, n_ne])
        triangles.append([n_sw, n_ne, n_nw])
    return np.asarray(triangles, dtype=int)


def plot_case_combined(
    mesh: MeshData,
    nodes: np.ndarray,
    result: CaseResult,
    output_dir: Path,
) -> None:
    """Save a single combined figure (mesh / displacement / Von Mises) for one case.

    Args:
        mesh: Original mesh data (corner nodes and connectivity).
        nodes: Node array used in this case (may include midside nodes for QH/TH).
        result: Solved case result.
        output_dir: Directory to save figures.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    fig.suptitle(f"Case {result.case_name}", fontsize=11)

    triangles = build_plot_triangles(mesh)
    triang_corner = Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], triangles=triangles)

    # Panel 0: Mesh preview
    ax0 = axes[0]
    ax0.set_title("Mesh")
    ax0.set_aspect("equal")
    ax0.triplot(triang_corner, color="0.25", linewidth=1.0)
    ax0.scatter(mesh.nodes[:, 0], mesh.nodes[:, 1], s=12, color="tab:red", zorder=3)
    ax0.set_xlabel("x [mm]")
    ax0.set_ylabel("y [mm]")
    ax0.set_xlim(-1.0, float(geometry["length"]) + 1.0)
    ax0.set_ylim(-0.8, float(geometry["height"]) + 0.8)

    # Panel 1: Displacement magnitude (Gouraud shading on enriched node set)
    ax1 = axes[1]
    ax1.set_title("Displacement magnitude [mm]")
    ax1.set_aspect("equal")
    disp_magnitude = np.linalg.norm(result.displacements.reshape(-1, 2), axis=1)
    triang_full = Triangulation(nodes[:, 0], nodes[:, 1], triangles=triangles)
    disp_interp = disp_magnitude[: nodes.shape[0]]
    tc1 = ax1.tripcolor(triang_full, disp_interp, shading="gouraud", cmap="viridis")
    plt.colorbar(tc1, ax=ax1)
    ax1.set_xlabel("x [mm]")
    ax1.set_ylabel("y [mm]")

    # Panel 2: Von Mises stress (Gouraud shading on nodal-averaged values)
    ax2 = axes[2]
    ax2.set_title("Von Mises stress [MPa]")
    ax2.set_aspect("equal")
    # nodal_vm is defined over mesh.nodes (corner + possible midside nodes);
    # triang_corner only references corner indices so only those values matter.
    nodal_vm = result.nodal_vm if result.nodal_vm is not None else np.zeros(mesh.nodes.shape[0])
    tc2 = ax2.tripcolor(triang_corner, nodal_vm[: mesh.nodes.shape[0]], shading="gouraud", cmap="hot_r")
    plt.colorbar(tc2, ax=ax2)
    ax2.set_xlabel("x [mm]")
    ax2.set_ylabel("y [mm]")

    fig.tight_layout()
    fname = output_dir / f"case_{result.case_name}.png"
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
        cell_w = mesh_pattern_cfg[res.mesh_pattern]["nx"]
        cell_w = L / cell_w
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
        "case,mesh_pattern,element_version,dof,"
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
            f"{r.case_name},{r.mesh_pattern},{r.element_version},{r.dof_count},"
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
    """Execute all currently implemented FEM cases and save outputs."""
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

    # Per-case combined plots (mesh + displacement + Von Mises in one figure)
    if output_options.get("plot_case_fields", True):
        for mesh, result, nodes_used in zip(meshes, results, nodes_per_case):
            plot_case_combined(mesh, nodes_used, result, output_dir)

    # Cross-section stress profiles
    plot_stress_profiles(results, analytical, output_dir)

    # Bar chart comparison
    plot_comparison_summary(results, analytical, output_dir)

    # CSV summary
    save_comparison(results, analytical, output_dir)

    # Console summary table
    header_fmt = f"{'Case':<12}{'Pattern':<8}{'Ver':<6}{'DOF':>6}  {'Tip[mm]':>10}  {'Err_T%':>8}  {'SxxMax':>8}  {'TauN':>8}  {'Balance':>10}  {'t[s]':>7}"
    LOGGER.info("Summary:\n%s", header_fmt)
    for r in results:
        ref_t = analytical.tip_deflection_timoshenko_mm
        err = abs(r.tip_deflection_mm - ref_t) / ref_t * 100.0 if ref_t else 0.0
        row = (
            f"{r.case_name:<12}{r.mesh_pattern:<8}{r.element_version:<6}{r.dof_count:>6}  "
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
