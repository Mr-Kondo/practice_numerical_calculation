# sec 2.1.2 — Cantilever Beam under Tip Shear (Plane Stress)

## 1. Problem description

A slender cantilever beam of length $L$, height $h$, and out-of-plane width $b$
is clamped at $x = 0$ and loaded by a uniform shear traction $\tau_0$ on the
right face ($x = L$).  The problem is analysed under **plane-stress** conditions
($\sigma_z = \tau_{xz} = \tau_{yz} = 0$).

Parameters used in this analysis:

| Symbol | Value | Unit |
|--------|-------|------|
| $E$ | 30 000 | MPa |
| $\nu$ | 0.30 | — |
| $L$ | 40 | mm |
| $h$ | 10 | mm |
| $b$ (thickness) | 1 | mm |
| $\tau_0$ | $-1$ | MPa (downward) |
| $P = \tau_0 \cdot h \cdot b$ | $10$ | N |

---

## 2. Analytical reference solution

### 2.1 Tip deflection

**Euler-Bernoulli** (bending only):

$$
\delta_{\text{EB}} = \frac{P L^3}{3 E I}
$$

**Timoshenko** (bending + shear):

$$
\delta_T = \frac{P L^3}{3 E I} + \frac{P L}{\kappa G A}
\qquad \kappa = \tfrac{5}{6} \text{ (rectangular section)}
$$

where $I = b h^3 / 12$, $A = b h$, $G = E / [2(1+\nu)]$.

For the given parameters:

$$
\delta_{\text{EB}} = 0.08533 \text{ mm}, \qquad \delta_T = 0.08949 \text{ mm}
$$

The shear contribution accounts for about 4.9 % of the total deflection.

### 2.2 Bending stress at the fixed end

$$
\sigma_x(x=0,\, y) = \frac{M \cdot z}{I},
\quad M = P L,
\quad z = y - \frac{h}{2}
$$

Peak value at $z = \pm h/2$:

$$
|\sigma_x|_{\max} = \frac{P L (h/2)}{I} = 24.0 \text{ MPa}
$$

### 2.3 Maximum shear stress

The parabolic distribution along any vertical cross-section:

$$
\tau_{xy}(y) = \frac{P}{2I}\left[\left(\frac{h}{2}\right)^2 - z^2\right]
$$

Peak at the neutral axis ($z = 0$):

$$
\tau_{xy,\max} = \frac{P h^2}{8 I} = \frac{3P}{2 A} = 1.5 \text{ MPa}
$$

---

## 3. Finite element formulations

### 3.1 Plane-stress constitutive matrix

$$
\mathbf{D} = \frac{E}{1-\nu^2}
\begin{bmatrix}
1 & \nu & 0 \\
\nu & 1 & 0 \\
0 & 0 & \tfrac{1-\nu}{2}
\end{bmatrix}
$$

### 3.2 CST element (3-node constant-strain triangle)

The strain field is constant within each element:

$$
\boldsymbol{\varepsilon} = \mathbf{B}\,\mathbf{u}^e,
\qquad \mathbf{B} = \frac{1}{2A}
\begin{bmatrix}
b_1 & 0 & b_2 & 0 & b_3 & 0 \\
0 & c_1 & 0 & c_2 & 0 & c_3 \\
c_1 & b_1 & c_2 & b_2 & c_3 & b_3
\end{bmatrix}
$$

Element stiffness: $\mathbf{K}^e = b A \,\mathbf{B}^T \mathbf{D} \mathbf{B}$.

### 3.3 Q4 element (4-node bilinear quadrilateral)

Shape functions in parent space $(\xi,\eta) \in [-1,1]^2$:

$$
N_i = \tfrac{1}{4}(1 + \xi_i\,\xi)(1 + \eta_i\,\eta)
$$

The stiffness is computed by $2 \times 2$ Gauss integration:

$$
\mathbf{K}^e = b \int_{-1}^{1}\!\int_{-1}^{1} \mathbf{B}^T \mathbf{D} \mathbf{B}
\det(\mathbf{J})\,d\xi\,d\eta
\approx b \sum_{k=1}^{4} \mathbf{B}_k^T \mathbf{D} \mathbf{B}_k \det(\mathbf{J}_k)
$$

---

## 4. Six mesh cases (figure 2.3)

| Case | Element | Mesh type | Split method | Nodes | DOF |
|------|---------|-----------|-------------|-------|-----|
| Q1 | Q4 | Regular coarse (4x2) | — | 15 | 30 |
| Q2 | Q4 | Distorted coarse (4x2) | — | 15 | 30 |
| Q3 | Q4 | Regular fine (8x4) | — | 45 | 90 |
| T1 | CST | Regular coarse (4x2) | X-split (4 tri/cell) | 23 | 46 |
| T2 | CST | Regular coarse (4x2) | Diagonal (2 tri/cell) | 15 | 30 |
| T3 | CST | Regular fine (8x4) | X-split (4 tri/cell) | 77 | 154 |

### 4.1 Distorted mesh (Q2)

Interior node x-coordinates are shifted by a parallelogram distortion:

$$
\Delta x_j = \alpha \cdot \Delta x_{\text{cell}} \cdot \frac{j}{n_y}
$$

where $j$ is the row index, $\alpha = 0.5$.
Boundary nodes (left and right faces) are not shifted to preserve the exact
boundary conditions.  Distortion degrades the Jacobian condition and reduces
the effective order of the element.

### 4.2 X-split vs. diagonal split (T1 vs. T2)

**Diagonal split** (T2): each Q4 is cut by the SW-NE diagonal, producing
two triangles.  The pattern introduces directional bias: elements oriented
along one diagonal are stiffer for shear than elements oriented the other way.
This symmetry breaking leads to larger deflection underestimation.

**X-split** (T1, T3): each Q4 is subdivided into four triangles by inserting
a centroid node.  The four-way symmetry reduces directional bias and gives
better accuracy for a given number of Q4 parent cells, though at the cost
of extra DOFs (centroid nodes).

---

## 5. Results discussion

### 5.1 Deflection accuracy

| Case | Tip deflection [mm] | Error vs Timoshenko |
|------|--------------------:|--------------------:|
| Q1   | 0.06243 | 30.2 % |
| Q2   | 0.04802 | 46.3 % |
| Q3   | 0.08021 | 10.4 % |
| T1   | 0.05610 | 37.3 % |
| T2   | 0.03417 | 61.8 % |
| T3   | 0.07731 | 13.6 % |

Key observations:

- **Shear locking** is the dominant source of error on coarse meshes (Q1, T1, T2).
  Low-order displacement-based elements are too stiff in bending-dominated problems
  because spurious shear strains consume energy that should be stored as bending.
- **Q2 (distorted)** is worse than Q1 despite having the same DOF count.
  Mesh distortion degrades the Jacobian mapping and reduces integration accuracy.
- **Q3 (fine Q4)** is the best performer: 8 elements along the span reduce locking
  and give near-Timoshenko accuracy.
- **T2 (diagonal CST)** has the largest error because the diagonal split introduces
  strong directional bias that artificially stiffens the model.
- **T3 (fine X-split CST)** slightly outperforms Q3 in absolute error, but requires
  154 DOFs vs 90 for Q3, making Q3 more efficient per DOF.

### 5.2 Why CST underperforms Q4

CST has a constant strain field throughout each element.  For a beam in bending,
the strain varies linearly through the depth, so CST requires a much finer mesh
than Q4 to capture the same accuracy.  Q4, with its bilinear displacement field,
can represent the linear bending strain distribution much more faithfully.

### 5.3 Force balance verification

All cases achieve $|\text{balance error}| < 10^{-11}$, confirming that the assembly
and boundary-condition imposition are correct.

---

## 6. Output files

| File | Description |
|------|-------------|
| `outputs/case_<NAME>_fields.png` | Displacement magnitude + von Mises stress maps |
| `outputs/stress_profiles.png` | $\sigma_x$ at $x=0$ and $\tau_{xy}$ at $x=L/2$ for all cases vs. theory |
| `outputs/comparison_summary.png` | Bar charts: tip deflection, $|\sigma_x|_{\max}$, $\tau_{xy,\text{neutral}}$ |
| `outputs/comparison_metrics.csv` | Full numerical comparison table |

### 6.1 CSV columns

| Column | Description |
|--------|-------------|
| `case` | Case identifier (Q1–Q3, T1–T3) |
| `element_type` | q4 or cst |
| `mesh_type` | regular_coarse / distorted_coarse / regular_fine |
| `dof` | Total degrees of freedom |
| `tip_deflection_mm` | FEM tip deflection |
| `ref_EB_mm` | Euler-Bernoulli reference |
| `ref_timoshenko_mm` | Timoshenko reference |
| `error_vs_timoshenko_pct` | Percentage error relative to Timoshenko |
| `sigma_x_fixed_end_MPa` | FEM peak $|\sigma_x|$ near $x=0$ |
| `ref_sigma_x_MPa` | Beam theory $|\sigma_x|_{\max}$ at fixed end |
| `tau_xy_neutral_MPa` | FEM $\tau_{xy}$ near neutral axis at mid-span |
| `ref_tau_xy_MPa` | Beam theory parabolic peak shear |
| `max_von_mises_MPa` | Maximum von Mises stress across all elements |
| `force_balance_error` | $|\sum F_{\text{ext}} + \sum R| / |\sum F_{\text{ext}}|$ |
| `solve_time_s` | Wall-clock solve time in seconds |
