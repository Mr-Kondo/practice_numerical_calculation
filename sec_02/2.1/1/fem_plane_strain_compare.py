"""Plane-strain static FEM solver for a 1/2 rigid-frame model.

This script runs the same boundary-value problem with two element types:
- CST (3-node constant-strain triangle)
- Q4 (4-node bilinear quadrilateral with 2x2 Gauss integration)

Outputs are written to sec_02/2.1/1/outputs.
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
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

LOGGER = logging.getLogger(__name__)


material_props: dict[str, Any] = {
    "E": 30000.0,  # MPa == N/mm^2
    "nu": 0.30,
    "analysis_type": "plane_strain",
    "thickness": 1.0,  # mm
}

geometry: dict[str, Any] = {
    "width": 140.0,  # mm (1/2 model)
    "height": 120.0,  # mm
    "opening_x_min": 0.0,  # mm (touches symmetry plane in 1/2 model)
    "opening_x_max": 60.0,  # mm (half of 120 mm opening)
    "opening_height": 80.0,  # mm
    "reference_figure": "sec_02/2.1/1/fig/rigid-frame-structure.jpeg",
}

mesh_params: dict[str, int] = {
    "nx": 70,
    "ny": 60,
}

loads: dict[str, float] = {
    "top_traction_y": -1.0,  # N/mm^2 (downward)
}

boundary_conditions: dict[str, Any] = {
    "symmetry_x": 0.0,  # ux = 0 on this edge
    "anchor_mode": "bottom_right_uy_zero",  # suppress rigid-body y-translation
}

analysis_cases: list[dict[str, str]] = [
    {"name": "cst", "element_type": "cst"},
    {"name": "q4", "element_type": "q4"},
]

output_options: dict[str, Any] = {
    "output_dir": "sec_02/2.1/1/outputs",
    "displacement_scale": 8.0,
}

# Post-processing configuration for regional stress evaluation.
# Note: a point constraint (single-node Dirichlet condition, e.g. uy=0 at the anchor)
# can produce an artificial local stress concentration. This is a numerical artifact
# of the discrete boundary condition, not a physical feature of the structure.
# Evaluating the maximum stress while excluding the anchor neighbourhood provides
# a more reliable comparison between element formulations.
post_processing: dict[str, float] = {
    "anchor_exclusion_radius_mm": 10.0,  # exclude elements within this radius of anchor
    "opening_corner_radius_mm": 10.0,  # evaluate elements within this radius of opening corner
    "opening_corner_x": 60.0,  # x-coordinate of opening top-right corner [mm]
    "opening_corner_y": 80.0,  # y-coordinate of opening top-right corner [mm]
}


@dataclass
class MeshData:
    """Structured mesh and boundary metadata."""

    nodes: np.ndarray
    q4_elements: np.ndarray
    top_edges: np.ndarray
    symmetry_nodes: np.ndarray
    anchor_node: int


@dataclass
class CaseResult:
    """Solver outputs for a single element type."""

    case_name: str
    element_type: str
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
    # Regional stress metrics for post-processing interpretation.
    # max_von_mises_global equals max_von_mises; both are retained so that
    # existing comparisons continue to work alongside the new columns.
    max_von_mises_global: float
    max_von_mises_excluding_anchor: float
    max_von_mises_near_opening_corner: float


def plane_strain_matrix(young: float, poisson: float) -> np.ndarray:
    """Build constitutive matrix for isotropic plane strain.

    Args:
        young: Young's modulus in MPa.
        poisson: Poisson ratio.

    Returns:
        3x3 constitutive matrix in MPa units.
    """

    coef = young / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    return coef * np.array(
        [
            [1.0 - poisson, poisson, 0.0],
            [poisson, 1.0 - poisson, 0.0],
            [0.0, 0.0, 0.5 * (1.0 - 2.0 * poisson)],
        ],
        dtype=float,
    )


def generate_mesh(geom: dict[str, Any], mesh_cfg: dict[str, int]) -> MeshData:
    """Generate a structured mesh for outer rectangle minus lower opening."""

    width = float(geom["width"])
    height = float(geom["height"])
    ox_min = float(geom["opening_x_min"])
    ox_max = float(geom["opening_x_max"])
    oy_max = float(geom["opening_height"])

    nx = int(mesh_cfg["nx"])
    ny = int(mesh_cfg["ny"])

    x_coords = np.linspace(0.0, width, nx + 1)
    y_coords = np.linspace(0.0, height, ny + 1)
    xx, yy = np.meshgrid(x_coords, y_coords, indexing="xy")
    nodes = np.column_stack((xx.ravel(), yy.ravel()))

    def node_id(i: int, j: int) -> int:
        return j * (nx + 1) + i

    q4_elements: list[list[int]] = []
    top_edges: list[list[int]] = []

    def cell_in_opening(i: int, j: int) -> bool:
        x_center = 0.5 * (x_coords[i] + x_coords[i + 1])
        y_center = 0.5 * (y_coords[j] + y_coords[j + 1])
        return (ox_min <= x_center <= ox_max) and (0.0 <= y_center <= oy_max)

    for j in range(ny):
        for i in range(nx):
            if cell_in_opening(i, j):
                continue
            n1 = node_id(i, j)
            n2 = node_id(i + 1, j)
            n3 = node_id(i, j + 1)
            n4 = node_id(i + 1, j + 1)
            q4_elements.append([n1, n2, n4, n3])

    # Top boundary segments for uniform traction.
    for i in range(nx):
        j = ny - 1
        if cell_in_opening(i, j):
            continue
        top_edges.append([node_id(i, ny), node_id(i + 1, ny)])

    if not q4_elements:
        raise ValueError("No active elements were generated. Check geometry and mesh resolution.")

    used_nodes = np.unique(np.asarray(q4_elements, dtype=int).ravel())
    remap = -np.ones(nodes.shape[0], dtype=int)
    remap[used_nodes] = np.arange(used_nodes.size, dtype=int)

    compact_nodes = nodes[used_nodes]
    compact_q4 = remap[np.asarray(q4_elements, dtype=int)]
    compact_top_edges = remap[np.asarray(top_edges, dtype=int)]

    symmetry_x = float(boundary_conditions["symmetry_x"])
    symmetry_nodes = np.where(np.isclose(compact_nodes[:, 0], symmetry_x))[0]

    bottom_nodes = np.where(np.isclose(compact_nodes[:, 1], 0.0))[0]
    if boundary_conditions.get("anchor_mode") == "bottom_right_uy_zero" and bottom_nodes.size > 0:
        anchor_node = int(bottom_nodes[np.argmax(compact_nodes[bottom_nodes, 0])])
    elif bottom_nodes.size > 0:
        anchor_node = int(bottom_nodes[0])
    elif symmetry_nodes.size > 0:
        anchor_node = int(symmetry_nodes[0])
    else:
        anchor_node = 0

    return MeshData(
        nodes=compact_nodes,
        q4_elements=compact_q4,
        top_edges=compact_top_edges,
        symmetry_nodes=symmetry_nodes.astype(int),
        anchor_node=anchor_node,
    )


def q4_to_cst(q4_elements: np.ndarray) -> np.ndarray:
    """Split each Q4 into two triangles with positive orientation."""

    tri_list: list[list[int]] = []
    for n1, n2, n4, n3 in q4_elements:
        tri_list.append([n1, n2, n4])
        tri_list.append([n1, n4, n3])
    return np.asarray(tri_list, dtype=int)


def cst_stiffness(coords: np.ndarray, constitutive: np.ndarray, thickness: float) -> np.ndarray:
    """Compute CST element stiffness matrix."""

    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]

    two_area = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    area = 0.5 * abs(two_area)
    if area <= 0.0:
        raise ValueError("Non-positive CST element area encountered.")

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

    return thickness * area * (b_mat.T @ constitutive @ b_mat)


def q4_shape_derivatives(xi: float, eta: float) -> np.ndarray:
    """Return derivatives dN/dxi and dN/deta for Q4 in parent coordinates."""

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
    """Build Q4 strain-displacement matrix and Jacobian determinant."""

    dnd_parent = q4_shape_derivatives(xi, eta)
    jacobian = coords.T @ dnd_parent
    det_j = float(np.linalg.det(jacobian))
    if det_j <= 0.0:
        raise ValueError("Invalid Q4 Jacobian determinant (<= 0).")

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
    """Compute Q4 element stiffness matrix with 2x2 Gauss integration."""

    gauss = 1.0 / np.sqrt(3.0)
    gauss_points = [(-gauss, -gauss), (gauss, -gauss), (gauss, gauss), (-gauss, gauss)]

    ke = np.zeros((8, 8), dtype=float)
    for xi, eta in gauss_points:
        b_mat, det_j = q4_b_matrix(coords, xi, eta)
        ke += thickness * (b_mat.T @ constitutive @ b_mat) * det_j
    return ke


def assemble_system(
    mesh: MeshData,
    element_type: str,
    constitutive: np.ndarray,
    thickness: float,
    traction_y: float,
) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    """Assemble global stiffness matrix and force vector."""

    if element_type == "cst":
        elements = q4_to_cst(mesh.q4_elements)
        dof_per_element = 6
    elif element_type == "q4":
        elements = mesh.q4_elements
        dof_per_element = 8
    else:
        raise ValueError(f"Unsupported element_type: {element_type}")

    ndof = 2 * mesh.nodes.shape[0]
    stiffness = lil_matrix((ndof, ndof), dtype=float)
    force = np.zeros(ndof, dtype=float)

    for conn in elements:
        coords = mesh.nodes[conn]
        if element_type == "cst":
            ke = cst_stiffness(coords, constitutive, thickness)
        else:
            ke = q4_stiffness(coords, constitutive, thickness)

        dofs = np.zeros(dof_per_element, dtype=int)
        for local_idx, node_id in enumerate(conn):
            dofs[2 * local_idx] = 2 * node_id
            dofs[2 * local_idx + 1] = 2 * node_id + 1

        for i_local, i_global in enumerate(dofs):
            for j_local, j_global in enumerate(dofs):
                stiffness[i_global, j_global] += ke[i_local, j_local]

    # Apply top-edge traction as equivalent nodal forces.
    for n1, n2 in mesh.top_edges:
        x1, y1 = mesh.nodes[n1]
        x2, y2 = mesh.nodes[n2]
        edge_length = float(np.hypot(x2 - x1, y2 - y1))
        nodal_fy = traction_y * thickness * edge_length / 2.0
        force[2 * n1 + 1] += nodal_fy
        force[2 * n2 + 1] += nodal_fy

    return stiffness.tocsr(), force, elements


def apply_boundary_conditions(mesh: MeshData) -> tuple[np.ndarray, np.ndarray]:
    """Build constrained/free DOF sets."""

    fixed_dofs: list[int] = []

    for node_id in mesh.symmetry_nodes:
        fixed_dofs.append(2 * int(node_id))  # ux = 0 on symmetry edge

    fixed_dofs.append(2 * mesh.anchor_node + 1)  # uy = 0 at anchor node

    fixed = np.unique(np.asarray(fixed_dofs, dtype=int))
    all_dofs = np.arange(2 * mesh.nodes.shape[0], dtype=int)
    free = np.setdiff1d(all_dofs, fixed)
    return fixed, free


def solve_case(
    mesh: MeshData,
    case_cfg: dict[str, str],
    constitutive: np.ndarray,
    anchor_coords: np.ndarray,
) -> CaseResult:
    """Solve one FEM case and compute stress/reaction metrics."""

    element_type = case_cfg["element_type"]
    case_name = case_cfg["name"]

    thickness = float(material_props["thickness"])
    traction_y = float(loads["top_traction_y"])

    start = time.perf_counter()
    stiffness, force, elements = assemble_system(mesh, element_type, constitutive, thickness, traction_y)
    fixed, free = apply_boundary_conditions(mesh)

    displacements = np.zeros(force.shape[0], dtype=float)
    kff = stiffness[free][:, free]
    ff = force[free]

    displacements[free] = spsolve(kff, ff)
    reactions = stiffness @ displacements - force
    elapsed = time.perf_counter() - start

    stress, centers, von_mises = compute_element_stress(mesh.nodes, elements, displacements, constitutive, element_type)

    vm_global, vm_excl_anchor, vm_near_corner = compute_regional_stress_metrics(
        centers, von_mises, anchor_coords, post_processing
    )

    uy_reaction = reactions[2 * mesh.anchor_node + 1]
    external_fy = float(np.sum(force[1::2]))
    balance_error = abs(external_fy + uy_reaction) / max(abs(external_fy), 1.0)

    nodal_u = displacements.reshape(-1, 2)
    max_disp = float(np.max(np.linalg.norm(nodal_u, axis=1)))

    return CaseResult(
        case_name=case_name,
        element_type=element_type,
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
        max_von_mises_global=vm_global,
        max_von_mises_excluding_anchor=vm_excl_anchor,
        max_von_mises_near_opening_corner=vm_near_corner,
    )


def von_mises_plane_strain(sigma_x: float, sigma_y: float, tau_xy: float, poisson: float) -> float:
    """Compute von Mises stress for plane-strain state."""

    sigma_z = poisson * (sigma_x + sigma_y)
    term = 0.5 * ((sigma_x - sigma_y) ** 2 + (sigma_y - sigma_z) ** 2 + (sigma_z - sigma_x) ** 2)
    return float(np.sqrt(term + 3.0 * tau_xy * tau_xy))


def cst_b_matrix(coords: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute CST B matrix and area."""

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


def compute_element_stress(
    nodes: np.ndarray,
    elements: np.ndarray,
    displacements: np.ndarray,
    constitutive: np.ndarray,
    element_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute element stress and von Mises values."""

    poisson = float(material_props["nu"])
    elem_stress: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    vm_list: list[float] = []

    for conn in elements:
        coords = nodes[conn]
        ue = np.zeros(2 * conn.size, dtype=float)
        for local_i, node_id in enumerate(conn):
            ue[2 * local_i] = displacements[2 * node_id]
            ue[2 * local_i + 1] = displacements[2 * node_id + 1]

        if element_type == "cst":
            b_mat, _ = cst_b_matrix(coords)
            strain = b_mat @ ue
            sigma = constitutive @ strain
        else:
            gauss = 1.0 / np.sqrt(3.0)
            points = [(-gauss, -gauss), (gauss, -gauss), (gauss, gauss), (-gauss, gauss)]
            sigmas = []
            for xi, eta in points:
                b_mat, _ = q4_b_matrix(coords, xi, eta)
                strain = b_mat @ ue
                sigmas.append(constitutive @ strain)
            sigma = np.mean(np.asarray(sigmas), axis=0)

        sxx = float(sigma[0])
        syy = float(sigma[1])
        txy = float(sigma[2])
        vm = von_mises_plane_strain(sxx, syy, txy, poisson)

        elem_stress.append(np.array([sxx, syy, txy], dtype=float))
        centers.append(np.mean(coords, axis=0))
        vm_list.append(vm)

    return np.asarray(elem_stress), np.asarray(centers), np.asarray(vm_list)


def compute_regional_stress_metrics(
    element_centers: np.ndarray,
    von_mises: np.ndarray,
    anchor_coords: np.ndarray,
    post_cfg: dict[str, float],
) -> tuple[float, float, float]:
    """Compute regional von Mises stress metrics for post-processing evaluation.

    Point constraints (single-node Dirichlet conditions such as the uy=0 anchor)
    can produce artificial local stress concentrations. These are a numerical
    artifact of the discrete boundary condition, not a physical feature of the
    structure. CST and Q4 handle this artifact differently, so the raw global
    maximum may reflect element-type sensitivity rather than structural behaviour.

    Evaluating the maximum stress while excluding the anchor neighbourhood, and
    additionally evaluating stress near the opening corner, gives a more
    physically meaningful comparison between element formulations.

    Args:
        element_centers: Array of shape (N, 2) with element centroid coordinates.
        von_mises: Array of shape (N,) with element von Mises stresses in MPa.
        anchor_coords: Shape (2,) array with anchor node (x, y) coordinates in mm.
        post_cfg: Post-processing configuration dict with keys:
            - anchor_exclusion_radius_mm
            - opening_corner_radius_mm
            - opening_corner_x
            - opening_corner_y

    Returns:
        Tuple of (vm_global_max, vm_excl_anchor_max, vm_near_corner_max) in MPa.
        vm_near_corner_max is 0.0 if no elements fall within the evaluation radius.
    """
    excl_radius = float(post_cfg["anchor_exclusion_radius_mm"])
    corner_radius = float(post_cfg["opening_corner_radius_mm"])
    corner_x = float(post_cfg["opening_corner_x"])
    corner_y = float(post_cfg["opening_corner_y"])

    # Global maximum over all elements (may include anchor stress artifact).
    vm_global_max = float(np.max(von_mises))

    # Distance from each element centre to the anchor node.
    # Elements inside the exclusion radius are omitted from the comparison
    # because the point constraint (uy=0) produces an artificial stress peak.
    dist_to_anchor = np.linalg.norm(element_centers - anchor_coords, axis=1)
    mask_excl = dist_to_anchor > excl_radius
    if np.any(mask_excl):
        vm_excl_anchor_max = float(np.max(von_mises[mask_excl]))
    else:
        LOGGER.warning(
            "All elements fall within anchor exclusion radius (%.1f mm). Returning global max for anchor-excluded metric.",
            excl_radius,
        )
        vm_excl_anchor_max = vm_global_max

    # Distance from each element centre to the opening top-right corner.
    # Only elements within the evaluation radius are considered.
    corner_coords = np.array([corner_x, corner_y], dtype=float)
    dist_to_corner = np.linalg.norm(element_centers - corner_coords, axis=1)
    mask_corner = dist_to_corner <= corner_radius
    if np.any(mask_corner):
        vm_near_corner_max = float(np.max(von_mises[mask_corner]))
    else:
        LOGGER.warning(
            "No elements found within opening corner evaluation radius (%.1f mm) around (%.1f, %.1f). Returning 0.0.",
            corner_radius,
            corner_x,
            corner_y,
        )
        vm_near_corner_max = 0.0

    return vm_global_max, vm_excl_anchor_max, vm_near_corner_max


def build_plot_triangles(elements: np.ndarray, element_type: str) -> tuple[np.ndarray, np.ndarray]:
    """Convert element connectivity into plotting triangles and face map."""

    if element_type == "cst":
        tri_conn = elements.copy()
        elem_index = np.arange(elements.shape[0], dtype=int)
        return tri_conn, elem_index

    tri_list: list[list[int]] = []
    elem_map: list[int] = []
    for eidx, (n1, n2, n4, n3) in enumerate(elements):
        tri_list.append([n1, n2, n4])
        tri_list.append([n1, n4, n3])
        elem_map.extend([eidx, eidx])
    return np.asarray(tri_list, dtype=int), np.asarray(elem_map, dtype=int)


def save_nodal_csv(nodes: np.ndarray, displacements: np.ndarray, path: Path) -> None:
    """Save nodal displacement results."""

    nodal_u = displacements.reshape(-1, 2)
    umag = np.linalg.norm(nodal_u, axis=1)

    with path.open("w", encoding="utf-8", newline="") as fobj:
        writer = csv.writer(fobj)
        writer.writerow(["node_id", "x_mm", "y_mm", "ux_mm", "uy_mm", "u_mag_mm"])
        for idx, (xy, uv, mag) in enumerate(zip(nodes, nodal_u, umag)):
            writer.writerow([idx, float(xy[0]), float(xy[1]), float(uv[0]), float(uv[1]), float(mag)])


def save_element_csv(stress: np.ndarray, centers: np.ndarray, von_mises: np.ndarray, path: Path) -> None:
    """Save element stress outputs."""

    with path.open("w", encoding="utf-8", newline="") as fobj:
        writer = csv.writer(fobj)
        writer.writerow(["element_id", "cx_mm", "cy_mm", "sigma_x_MPa", "sigma_y_MPa", "tau_xy_MPa", "von_mises_MPa"])
        for eidx, (sig, ctr, vm) in enumerate(zip(stress, centers, von_mises)):
            writer.writerow([eidx, float(ctr[0]), float(ctr[1]), float(sig[0]), float(sig[1]), float(sig[2]), float(vm)])


def plot_case(mesh: MeshData, result: CaseResult, output_dir: Path) -> None:
    """Plot displacement magnitude and von Mises stress for one case."""

    nodes = mesh.nodes
    nodal_u = result.displacements.reshape(-1, 2)
    umag = np.linalg.norm(nodal_u, axis=1)

    tri_conn, elem_map = build_plot_triangles(result.element_conn, result.element_type)
    tri = Triangulation(nodes[:, 0], nodes[:, 1], triangles=tri_conn)

    # Displacement plot (undeformed + deformed).
    max_disp = max(float(np.max(umag)), 1.0e-12)
    scale = float(output_options["displacement_scale"]) * max(geometry["width"], geometry["height"]) / max_disp / 50.0
    deformed = nodes + scale * nodal_u

    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    contour = ax.tricontourf(tri, umag, levels=24, cmap="viridis")
    ax.triplot(nodes[:, 0], nodes[:, 1], tri_conn, color="0.75", linewidth=0.4, alpha=0.6)
    ax.triplot(deformed[:, 0], deformed[:, 1], tri_conn, color="black", linewidth=0.5, alpha=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Displacement magnitude ({result.case_name.upper()})")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    fig.colorbar(contour, ax=ax, label="|u| [mm]")
    fig.tight_layout()
    fig.savefig(output_dir / f"displacement_{result.case_name}.png")
    plt.close(fig)

    # von Mises plot using element-based face colors.
    face_values = result.von_mises[elem_map]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    poly = ax.tripcolor(tri, facecolors=face_values, cmap="plasma", edgecolors="k", linewidth=0.08)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Von Mises stress ({result.case_name.upper()})")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    fig.colorbar(poly, ax=ax, label="von Mises [MPa]")
    fig.tight_layout()
    fig.savefig(output_dir / f"von_mises_{result.case_name}.png")
    plt.close(fig)

    # Mirrored full-model view for interpretation of the symmetry solution.
    nodes_mirror = nodes.copy()
    nodes_mirror[:, 0] *= -1.0
    u_mirror = nodal_u.copy()
    u_mirror[:, 0] *= -1.0

    full_nodes = np.vstack([nodes_mirror, nodes])
    full_u = np.vstack([u_mirror, nodal_u])
    full_umag = np.linalg.norm(full_u, axis=1)

    node_offset = nodes.shape[0]
    tri_mirror = tri_conn.copy()
    tri_mirror = tri_mirror[:, [0, 2, 1]]
    tri_full = np.vstack([tri_mirror, tri_conn + node_offset])
    tri_full_obj = Triangulation(full_nodes[:, 0], full_nodes[:, 1], triangles=tri_full)

    max_disp_full = max(float(np.max(full_umag)), 1.0e-12)
    scale_full = (
        float(output_options["displacement_scale"]) * max(geometry["width"], geometry["height"]) / max_disp_full / 50.0
    )
    full_deformed = full_nodes + scale_full * full_u

    fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
    contour = ax.tricontourf(tri_full_obj, full_umag, levels=24, cmap="viridis")
    ax.triplot(full_nodes[:, 0], full_nodes[:, 1], tri_full, color="0.75", linewidth=0.35, alpha=0.6)
    ax.triplot(full_deformed[:, 0], full_deformed[:, 1], tri_full, color="black", linewidth=0.45, alpha=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Displacement magnitude (mirrored full model, {result.case_name.upper()})")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    fig.colorbar(contour, ax=ax, label="|u| [mm]")
    fig.tight_layout()
    fig.savefig(output_dir / f"displacement_{result.case_name}_full_mirror.png")
    plt.close(fig)

    full_face_values = np.concatenate([face_values, face_values])
    fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
    poly = ax.tripcolor(tri_full_obj, facecolors=full_face_values, cmap="plasma", edgecolors="k", linewidth=0.06)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Von Mises stress (mirrored full model, {result.case_name.upper()})")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    fig.colorbar(poly, ax=ax, label="von Mises [MPa]")
    fig.tight_layout()
    fig.savefig(output_dir / f"von_mises_{result.case_name}_full_mirror.png")
    plt.close(fig)


def save_comparison(results: list[CaseResult], output_dir: Path) -> None:
    """Save case comparison CSV and summary chart."""

    csv_path = output_dir / "comparison_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fobj:
        writer = csv.writer(fobj)
        writer.writerow(
            [
                "case",
                "element_type",
                "max_disp_mm",
                "max_von_mises_MPa",
                "max_von_mises_global_MPa",
                "max_von_mises_excluding_anchor_MPa",
                "max_von_mises_near_opening_corner_MPa",
                "anchor_exclusion_radius_mm",
                "opening_corner_radius_mm",
                "force_balance_error",
                "dof",
                "solve_time_s",
            ]
        )
        for item in results:
            writer.writerow(
                [
                    item.case_name,
                    item.element_type,
                    item.max_disp,
                    item.max_von_mises,
                    item.max_von_mises_global,
                    item.max_von_mises_excluding_anchor,
                    item.max_von_mises_near_opening_corner,
                    post_processing["anchor_exclusion_radius_mm"],
                    post_processing["opening_corner_radius_mm"],
                    item.force_balance_error,
                    item.dof_count,
                    item.solve_time_s,
                ]
            )

    labels = [item.case_name.upper() for item in results]
    max_disp = [item.max_disp for item in results]
    max_vm = [item.max_von_mises for item in results]
    balance = [item.force_balance_error for item in results]
    dof_vals = [item.dof_count for item in results]
    solve_t = [item.solve_time_s for item in results]

    fig, axes = plt.subplots(2, 3, figsize=(11, 6), dpi=140)
    metrics = [
        (max_disp, "Max displacement [mm]"),
        (max_vm, "Max von Mises [MPa]"),
        (balance, "Force balance error [-]"),
        (dof_vals, "DOF count"),
        (solve_t, "Solve time [s]"),
    ]

    for ax, (vals, title) in zip(axes.ravel(), metrics):
        ax.bar(labels, vals, color=["#4C78A8", "#F58518"])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    axes.ravel()[-1].axis("off")
    fig.suptitle("CST vs Q4 comparison", y=0.98)
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_summary.png")
    plt.close(fig)


def run() -> None:
    """Execute both FEM cases and save all outputs."""

    output_dir = Path(output_options["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh = generate_mesh(geometry, mesh_params)
    constitutive = plane_strain_matrix(float(material_props["E"]), float(material_props["nu"]))

    LOGGER.info("Reference figure: %s", geometry["reference_figure"])
    LOGGER.info(
        "Mesh generated: nodes=%d, q4_elements=%d, top_edges=%d",
        mesh.nodes.shape[0],
        mesh.q4_elements.shape[0],
        mesh.top_edges.shape[0],
    )

    anchor_coords = mesh.nodes[mesh.anchor_node]
    LOGGER.info(
        "Anchor node index=%d, coordinates=(%.3f, %.3f) mm  [point constraint uy=0; may produce artificial local stress peak]",
        mesh.anchor_node,
        float(anchor_coords[0]),
        float(anchor_coords[1]),
    )

    results: list[CaseResult] = []
    for case in analysis_cases:
        LOGGER.info("Running case: %s", case["name"])
        result = solve_case(mesh, case, constitutive, anchor_coords)
        results.append(result)

        save_nodal_csv(mesh.nodes, result.displacements, output_dir / f"nodal_displacement_{result.case_name}.csv")
        save_element_csv(
            result.element_stress,
            result.element_centers,
            result.von_mises,
            output_dir / f"element_stress_{result.case_name}.csv",
        )
        np.savez(
            output_dir / f"results_{result.case_name}.npz",
            nodes=mesh.nodes,
            elements=result.element_conn,
            displacements=result.displacements.reshape(-1, 2),
            stress=result.element_stress,
            centers=result.element_centers,
            von_mises=result.von_mises,
        )
        plot_case(mesh, result, output_dir)

        LOGGER.info(
            "Case %s done: max_disp=%.6e mm, max_vm=%.6e MPa, balance_error=%.3e, solve_time=%.3fs",
            result.case_name,
            result.max_disp,
            result.max_von_mises,
            result.force_balance_error,
            result.solve_time_s,
        )

    save_comparison(results, output_dir)


def main() -> None:
    """CLI entry point with robust error reporting."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    try:
        run()
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("FEM execution failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
