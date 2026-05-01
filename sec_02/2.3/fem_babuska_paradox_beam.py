"""Reproduce Babuska's paradox with point supports and a point load.

This script compares two Q4 (bilinear quadrilateral) plane-stress models
for the same simply supported beam-like rectangular domain:

- Qa: globally coarse mesh
- Qb: aggressive local refinement near point supports and point load

The purpose is to compare deformation shapes and highlight mesh-sensitivity
caused by singular point boundary conditions.
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
    "analysis_type": "plane_stress",
    "thickness": 10.0,  # mm (beam depth in out-of-plane direction)
}

geometry: dict[str, Any] = {
    "length": 120.0,  # mm
    "height": 20.0,  # mm
    "reference_figure": "sec_02/2.3/fig/Babuska_paradox.png",
}

loads: dict[str, float] = {
    "point_load_fy": -100.0,  # N (downward)
}

analysis_cases: list[dict[str, str]] = [
    {"name": "Qa", "mesh_type": "coarse_uniform"},
    {"name": "Qb", "mesh_type": "locally_refined"},
]

mesh_params: dict[str, Any] = {
    "qa": {
        "nx": 24,
        "ny": 4,
    },
    "qb": {
        "base_nx": 24,
        "base_ny": 4,
        "x_clusters": [
            {"center": 0.0, "radius": 12.0, "spacing": 1.0},
            {"center": 60.0, "radius": 12.0, "spacing": 1.0},
            {"center": 120.0, "radius": 12.0, "spacing": 1.0},
        ],
        "y_clusters": [
            {"center": 0.0, "radius": 4.0, "spacing": 0.5},
            {"center": 20.0, "radius": 4.0, "spacing": 0.5},
        ],
    },
}

output_options: dict[str, Any] = {
    "output_dir": "sec_02/2.3/outputs",
    "deformation_scale_factor": 2.0,
}


@dataclass
class MeshData:
    """Structured mesh and boundary metadata for one case."""

    case_name: str
    mesh_type: str
    nodes: np.ndarray
    q4_elements: np.ndarray
    support_left_node: int
    support_right_node: int
    load_node: int


@dataclass
class CaseResult:
    """Solver outputs for one mesh case."""

    case_name: str
    mesh_type: str
    solve_time_s: float
    displacements: np.ndarray
    reactions: np.ndarray
    external_force_vector: np.ndarray
    dof_count: int
    max_disp: float
    force_balance_error: float
    support_reactions: dict[str, float]


def plane_stress_matrix(young: float, poisson: float) -> np.ndarray:
    """Build isotropic constitutive matrix for plane stress.

    Args:
        young: Young's modulus in MPa.
        poisson: Poisson ratio.

    Returns:
        3x3 constitutive matrix.
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


def _unique_sorted(values: np.ndarray, decimals: int = 10) -> np.ndarray:
    """Return sorted unique values with rounding to suppress float duplicates."""

    rounded = np.round(values.astype(float), decimals=decimals)
    unique = np.unique(rounded)
    unique.sort()
    return unique


def build_uniform_axis(length: float, divisions: int) -> np.ndarray:
    """Build a uniform coordinate axis including both endpoints."""

    if divisions <= 0:
        raise ValueError("divisions must be positive")
    return np.linspace(0.0, length, divisions + 1, dtype=float)


def build_clustered_axis(
    length: float,
    base_divisions: int,
    clusters: list[dict[str, float]],
    required_points: list[float],
) -> np.ndarray:
    """Build non-uniform axis with local refinement around target coordinates.

    Args:
        length: Axis total length.
        base_divisions: Number of base uniform divisions.
        clusters: List of dicts with center, radius, spacing.
        required_points: Coordinates that must exist on the axis.

    Returns:
        Sorted unique coordinate array.
    """

    coords: list[np.ndarray] = [build_uniform_axis(length, base_divisions)]

    for item in clusters:
        center = float(item["center"])
        radius = float(item["radius"])
        spacing = float(item["spacing"])
        left = max(0.0, center - radius)
        right = min(length, center + radius)
        if right <= left:
            continue

        npoint = int(np.ceil((right - left) / spacing)) + 1
        npoint = max(npoint, 2)
        coords.append(np.linspace(left, right, npoint, dtype=float))

    if required_points:
        coords.append(np.asarray(required_points, dtype=float))

    merged = np.concatenate(coords)
    return _unique_sorted(merged)


def closest_node_index(nodes: np.ndarray, x: float, y: float) -> int:
    """Return nearest node index for a target coordinate."""

    target = np.array([x, y], dtype=float)
    d2 = np.sum((nodes - target) ** 2, axis=1)
    return int(np.argmin(d2))


def generate_mesh(case_name: str, mesh_type: str) -> MeshData:
    """Generate Qa or Qb mesh for the beam domain.

    Args:
        case_name: Display name of the case.
        mesh_type: coarse_uniform or locally_refined.

    Returns:
        MeshData for FEM assembly.
    """

    length = float(geometry["length"])
    height = float(geometry["height"])

    if mesh_type == "coarse_uniform":
        x_coords = build_uniform_axis(length, int(mesh_params["qa"]["nx"]))
        y_coords = build_uniform_axis(height, int(mesh_params["qa"]["ny"]))
    elif mesh_type == "locally_refined":
        qbcfg = mesh_params["qb"]
        x_coords = build_clustered_axis(
            length=length,
            base_divisions=int(qbcfg["base_nx"]),
            clusters=list(qbcfg["x_clusters"]),
            required_points=[0.0, length / 2.0, length],
        )
        y_coords = build_clustered_axis(
            length=height,
            base_divisions=int(qbcfg["base_ny"]),
            clusters=list(qbcfg["y_clusters"]),
            required_points=[0.0, height],
        )
    else:
        raise ValueError(f"Unsupported mesh_type: {mesh_type}")

    nx = x_coords.size - 1
    ny = y_coords.size - 1

    xx, yy = np.meshgrid(x_coords, y_coords, indexing="xy")
    nodes = np.column_stack((xx.ravel(), yy.ravel()))

    def node_id(i: int, j: int) -> int:
        return j * (nx + 1) + i

    q4_elements: list[list[int]] = []
    for j in range(ny):
        for i in range(nx):
            n1 = node_id(i, j)
            n2 = node_id(i + 1, j)
            n3 = node_id(i, j + 1)
            n4 = node_id(i + 1, j + 1)
            q4_elements.append([n1, n2, n4, n3])

    q4 = np.asarray(q4_elements, dtype=int)
    if q4.size == 0:
        raise ValueError("No Q4 elements generated")

    support_left_node = closest_node_index(nodes, 0.0, 0.0)
    support_right_node = closest_node_index(nodes, length, 0.0)
    load_node = closest_node_index(nodes, length / 2.0, height)

    return MeshData(
        case_name=case_name,
        mesh_type=mesh_type,
        nodes=nodes,
        q4_elements=q4,
        support_left_node=support_left_node,
        support_right_node=support_right_node,
        load_node=load_node,
    )


def q4_shape_derivatives(xi: float, eta: float) -> np.ndarray:
    """Return derivatives dN/dxi and dN/deta for Q4 parent coordinates."""

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
    """Compute Q4 stiffness with 2x2 Gauss integration."""

    gauss = 1.0 / np.sqrt(3.0)
    gauss_points = [(-gauss, -gauss), (gauss, -gauss), (gauss, gauss), (-gauss, gauss)]

    ke = np.zeros((8, 8), dtype=float)
    for xi, eta in gauss_points:
        b_mat, det_j = q4_b_matrix(coords, xi, eta)
        ke += thickness * (b_mat.T @ constitutive @ b_mat) * det_j
    return ke


def assemble_system(mesh: MeshData, constitutive: np.ndarray) -> tuple[csr_matrix, np.ndarray]:
    """Assemble global stiffness and force vectors for one case."""

    ndof = 2 * mesh.nodes.shape[0]
    stiffness = lil_matrix((ndof, ndof), dtype=float)
    force = np.zeros(ndof, dtype=float)

    thickness = float(material_props["thickness"])

    for conn in mesh.q4_elements:
        coords = mesh.nodes[conn]
        ke = q4_stiffness(coords, constitutive, thickness)

        dofs = np.zeros(8, dtype=int)
        for local_idx, node_id in enumerate(conn):
            dofs[2 * local_idx] = 2 * node_id
            dofs[2 * local_idx + 1] = 2 * node_id + 1

        for i_local, i_global in enumerate(dofs):
            for j_local, j_global in enumerate(dofs):
                stiffness[i_global, j_global] += ke[i_local, j_local]

    # Point load at top center node.
    force[2 * mesh.load_node + 1] += float(loads["point_load_fy"])

    return stiffness.tocsr(), force


def build_dof_sets(mesh: MeshData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fixed and free DOFs and fixed y-DOFs for support reactions."""

    fixed_dofs = [
        2 * mesh.support_left_node,
        2 * mesh.support_left_node + 1,
        2 * mesh.support_right_node,
        2 * mesh.support_right_node + 1,
    ]
    fixed = np.unique(np.asarray(fixed_dofs, dtype=int))

    all_dofs = np.arange(2 * mesh.nodes.shape[0], dtype=int)
    free = np.setdiff1d(all_dofs, fixed)

    fixed_y = np.asarray([2 * mesh.support_left_node + 1, 2 * mesh.support_right_node + 1], dtype=int)
    return fixed, free, fixed_y


def solve_case(mesh: MeshData, constitutive: np.ndarray) -> CaseResult:
    """Solve one static FEM case and collect metrics."""

    start = time.perf_counter()
    stiffness, force = assemble_system(mesh, constitutive)
    fixed, free, fixed_y = build_dof_sets(mesh)

    displacements = np.zeros(force.shape[0], dtype=float)
    kff = stiffness[free][:, free]
    ff = force[free]

    displacements[free] = spsolve(kff, ff)
    reactions = stiffness @ displacements - force
    elapsed = time.perf_counter() - start

    nodal_u = displacements.reshape(-1, 2)
    max_disp = float(np.max(np.linalg.norm(nodal_u, axis=1)))

    external_fy = float(np.sum(force[1::2]))
    reaction_fy = float(np.sum(reactions[fixed_y]))
    balance_error = abs(external_fy + reaction_fy) / max(abs(external_fy), 1.0)

    support_reactions = {
        "rx_left": float(reactions[2 * mesh.support_left_node]),
        "ry_left": float(reactions[2 * mesh.support_left_node + 1]),
        "rx_right": float(reactions[2 * mesh.support_right_node]),
        "ry_right": float(reactions[2 * mesh.support_right_node + 1]),
    }

    return CaseResult(
        case_name=mesh.case_name,
        mesh_type=mesh.mesh_type,
        solve_time_s=elapsed,
        displacements=displacements,
        reactions=reactions,
        external_force_vector=force,
        dof_count=int(displacements.size),
        max_disp=max_disp,
        force_balance_error=float(balance_error),
        support_reactions=support_reactions,
    )


def quads_to_triangles(q4_elements: np.ndarray) -> np.ndarray:
    """Split each Q4 into two triangles for plotting."""

    tris: list[list[int]] = []
    for n1, n2, n4, n3 in q4_elements:
        tris.append([n1, n2, n4])
        tris.append([n1, n4, n3])
    return np.asarray(tris, dtype=int)


def save_nodal_csv(mesh: MeshData, result: CaseResult, output_dir: Path) -> None:
    """Write nodal displacement CSV."""

    path = output_dir / f"nodal_displacement_{result.case_name}.csv"
    nodal_u = result.displacements.reshape(-1, 2)
    umag = np.linalg.norm(nodal_u, axis=1)

    with path.open("w", encoding="utf-8", newline="") as fobj:
        writer = csv.writer(fobj)
        writer.writerow(["node_id", "x_mm", "y_mm", "ux_mm", "uy_mm", "u_mag_mm"])
        for i, (xy, uv, mag) in enumerate(zip(mesh.nodes, nodal_u, umag)):
            writer.writerow([i, float(xy[0]), float(xy[1]), float(uv[0]), float(uv[1]), float(mag)])


def save_case_npz(mesh: MeshData, result: CaseResult, output_dir: Path) -> None:
    """Write compact binary results for one case."""

    np.savez(
        output_dir / f"results_{result.case_name}.npz",
        nodes=mesh.nodes,
        q4_elements=mesh.q4_elements,
        displacements=result.displacements.reshape(-1, 2),
        reactions=result.reactions,
        load_node=mesh.load_node,
        support_left_node=mesh.support_left_node,
        support_right_node=mesh.support_right_node,
    )


def plot_case(mesh: MeshData, result: CaseResult, output_dir: Path, deformation_scale: float) -> None:
    """Save per-case displacement contour and deformed mesh overlay."""

    nodes = mesh.nodes
    nodal_u = result.displacements.reshape(-1, 2)
    umag = np.linalg.norm(nodal_u, axis=1)

    triangles = quads_to_triangles(mesh.q4_elements)
    tri = Triangulation(nodes[:, 0], nodes[:, 1], triangles=triangles)

    deformed = nodes + deformation_scale * nodal_u

    fig, ax = plt.subplots(figsize=(8, 3), dpi=150)
    contour = ax.tricontourf(tri, umag, levels=24, cmap="viridis")
    ax.triplot(nodes[:, 0], nodes[:, 1], triangles, color="0.75", linewidth=0.4, alpha=0.7)
    ax.triplot(deformed[:, 0], deformed[:, 1], triangles, color="black", linewidth=0.5, alpha=0.85)
    ax.scatter(
        [nodes[mesh.support_left_node, 0], nodes[mesh.support_right_node, 0], nodes[mesh.load_node, 0]],
        [nodes[mesh.support_left_node, 1], nodes[mesh.support_right_node, 1], nodes[mesh.load_node, 1]],
        s=30,
        c=["tab:red", "tab:red", "tab:blue"],
        marker="o",
        zorder=10,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{result.case_name}: displacement contour + deformed mesh")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    fig.colorbar(contour, ax=ax, label="|u| [mm]")
    fig.tight_layout()
    fig.savefig(output_dir / f"displacement_{result.case_name}.png")
    plt.close(fig)


def plot_comparison(
    meshes: dict[str, MeshData],
    results: dict[str, CaseResult],
    output_dir: Path,
    deformation_scale: float,
) -> None:
    """Save side-by-side Qa/Qb deformation comparison with common scaling."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), dpi=150)

    for ax, case_name in zip(axes, ["Qa", "Qb"]):
        mesh = meshes[case_name]
        result = results[case_name]

        nodes = mesh.nodes
        nodal_u = result.displacements.reshape(-1, 2)
        deformed = nodes + deformation_scale * nodal_u
        triangles = quads_to_triangles(mesh.q4_elements)

        ax.triplot(nodes[:, 0], nodes[:, 1], triangles, color="0.75", linewidth=0.4, alpha=0.8)
        ax.triplot(deformed[:, 0], deformed[:, 1], triangles, color="black", linewidth=0.5, alpha=0.9)
        ax.scatter(
            [nodes[mesh.support_left_node, 0], nodes[mesh.support_right_node, 0], nodes[mesh.load_node, 0]],
            [nodes[mesh.support_left_node, 1], nodes[mesh.support_right_node, 1], nodes[mesh.load_node, 1]],
            s=26,
            c=["tab:red", "tab:red", "tab:blue"],
            marker="o",
            zorder=8,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{case_name}  (max |u| = {result.max_disp:.4e} mm)")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")

    fig.suptitle(f"Babuska paradox comparison (common deformation scale = {deformation_scale:.1f})", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_deformation_Qa_Qb.png", bbox_inches="tight")
    plt.close(fig)


def save_comparison_csv(meshes: dict[str, MeshData], results: dict[str, CaseResult], output_dir: Path) -> None:
    """Write summary table for Qa/Qb comparison."""

    path = output_dir / "comparison_metrics.csv"
    with path.open("w", encoding="utf-8", newline="") as fobj:
        writer = csv.writer(fobj)
        writer.writerow([
            "case",
            "mesh_type",
            "nodes",
            "elements",
            "dof",
            "max_disp_mm",
            "force_balance_error",
            "rx_left_N",
            "ry_left_N",
            "rx_right_N",
            "ry_right_N",
            "solve_time_s",
        ])

        for case_name in ["Qa", "Qb"]:
            mesh = meshes[case_name]
            result = results[case_name]
            writer.writerow([
                case_name,
                result.mesh_type,
                mesh.nodes.shape[0],
                mesh.q4_elements.shape[0],
                result.dof_count,
                result.max_disp,
                result.force_balance_error,
                result.support_reactions["rx_left"],
                result.support_reactions["ry_left"],
                result.support_reactions["rx_right"],
                result.support_reactions["ry_right"],
                result.solve_time_s,
            ])


def run() -> None:
    """Run Qa/Qb simulations and save all outputs."""

    output_dir = Path(output_options["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    constitutive = plane_stress_matrix(float(material_props["E"]), float(material_props["nu"]))

    LOGGER.info("Reference figure: %s", geometry["reference_figure"])

    meshes: dict[str, MeshData] = {}
    results: dict[str, CaseResult] = {}

    for case in analysis_cases:
        case_name = str(case["name"])
        mesh_type = str(case["mesh_type"])

        mesh = generate_mesh(case_name=case_name, mesh_type=mesh_type)
        result = solve_case(mesh, constitutive)

        meshes[case_name] = mesh
        results[case_name] = result

        LOGGER.info(
            "%s solved: nodes=%d, elems=%d, dof=%d, max_disp=%.6e mm, balance=%.3e, time=%.3fs",
            case_name,
            mesh.nodes.shape[0],
            mesh.q4_elements.shape[0],
            result.dof_count,
            result.max_disp,
            result.force_balance_error,
            result.solve_time_s,
        )

        save_nodal_csv(mesh, result, output_dir)
        save_case_npz(mesh, result, output_dir)
        # Temporary scale; final side-by-side comparison uses shared scale.
        plot_case(mesh, result, output_dir, deformation_scale=1.0)

    global_max_disp = max(results["Qa"].max_disp, results["Qb"].max_disp, 1.0e-14)
    deformation_scale = float(output_options["deformation_scale_factor"]) * float(geometry["height"]) / global_max_disp

    # Re-render case plots with the same scale for apples-to-apples comparison.
    for case_name in ["Qa", "Qb"]:
        plot_case(meshes[case_name], results[case_name], output_dir, deformation_scale=deformation_scale)

    plot_comparison(meshes, results, output_dir, deformation_scale=deformation_scale)
    save_comparison_csv(meshes, results, output_dir)


def main() -> None:
    """CLI entry point."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    try:
        run()
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Babuska paradox run failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
