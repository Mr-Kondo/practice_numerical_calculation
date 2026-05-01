# Babuska paradox reproduction (Qa vs Qb, QL plane stress)

## 1. Purpose

This section reproduces the mesh-sensitivity known as Babuska's paradox
under point supports and a point load.

Two models are solved for the same geometry and material:

- Qa: globally coarse mesh
- Qb: local aggressive refinement around singular points only

Both use the same element family (QL: 4-node bilinear quadrilateral)
and plane-stress formulation.

## 2. Problem setup

- Geometry:
  - Length: 120 mm
  - Height: 20 mm
  - Out-of-plane thickness: 10 mm
- Material:
  - Young's modulus: 30000 MPa
  - Poisson ratio: 0.30
- Boundary conditions:
  - Point support at bottom-left node: ux = 0, uy = 0
  - Point support at bottom-right node: ux = 0, uy = 0
- Load:
  - Downward point load at top-center node: Fy = -100 N

## 3. Qa and Qb mesh design

- Qa:
  - Uniform coarse structured Q4 mesh.
- Qb:
  - Non-uniform structured Q4 mesh.
  - Coordinates are densely clustered near:
    - left support point,
    - right support point,
    - top-center load point.
  - The rest of the domain remains coarse.

This setting intentionally keeps singular point conditions while changing
local mesh density near those singularities.

## 4. Numerical formulation

The script solves linear static elasticity with plane stress:

$$
\mathbf{K}\mathbf{u}=\mathbf{f}
$$

- Constitutive matrix: isotropic plane stress
- Q4 stiffness: 2x2 Gauss integration
- Global solve: sparse direct solve (`scipy.sparse.linalg.spsolve`)

## 5. How to run

From repository root:

```bash
uv run python sec_02/2.3/fem_babuska_paradox_beam.py
```

## 6. Outputs

All outputs are written to `sec_02/2.3/outputs/`.

- `comparison_metrics.csv`
  - nodes, elements, DOF, max displacement, force-balance error,
    support reactions, solve time for Qa/Qb.
- `nodal_displacement_Qa.csv`, `nodal_displacement_Qb.csv`
  - nodal coordinates and displacement components.
- `results_Qa.npz`, `results_Qb.npz`
  - compact binary outputs for post-processing.
- `displacement_Qa.png`, `displacement_Qb.png`
  - displacement contour with undeformed/deformed mesh overlay.
- `comparison_deformation_Qa_Qb.png`
  - side-by-side Qa/Qb deformation plot with common scale.

## 7. What to inspect

When singular point constraints are present, local refinement can cause
an unintuitive deformation pattern near those points.

Inspect these points first:

- Deformation shape difference between Qa and Qb under the same scale.
- Whether Qb exhibits stronger local kinks near supports/load.
- Force-balance error magnitude in `comparison_metrics.csv`.

## 8. Notes

- This is a deliberate educational setup using point supports and point load.
- Singular conditions are known to induce mesh-sensitive local behavior.
- The purpose is comparison of qualitative behavior, not high-fidelity
  stress prediction at singular points.
