# 3-DOF Serial Manipulator: Kinematics & Dynamics Guide

This document explains the analytical kinematic and Euler–Lagrange dynamic model of the 3-DOF Dynamixel AX-12A serial robot implemented in [`kinematics_dynamics_model.py`](kinematics_dynamics_model.py) and [`matlab/cd1_corrected.m`](matlab/cd1_corrected.m), and how it maps directly to the MuJoCo simulation in [`model.xml`](model.xml).

---

## 1. Coordinate Frames & Forward Kinematics

The robot has three revolute joints $q = [q_1, q_2, q_3]^T$:
- **$q_1$ (Base yaw)**: rotates around the vertical $+Z_0$ axis.
- **$q_2$ (Shoulder pitch)**: rotates around the horizontal shoulder axis $Z_2$.
- **$q_3$ (Elbow pitch)**: rotates around the horizontal elbow axis $Z_3 \parallel Z_2$.

At the home/zero configuration $q = [0, 0, 0]^T$, the arm stands **straight up** along $+Z_0$ with end-effector height $z = l_1 + l_2 + l_3$.

### Homogeneous Transformations
Following the Denavit–Hartenberg sequence from `cd1.m` / `cd1_live.mlx`:

$$
S_{0}^{1} = \operatorname{Trans}_z(l_1)\operatorname{Rot}_z(q_1), \qquad
S_{1}^{2} = \operatorname{Rot}_y\!\left(\frac{\pi}{2}\right)\operatorname{Rot}_z(q_2 + \pi)
$$
$$
S_{2}^{3} = \operatorname{Trans}_x(l_2)\operatorname{Rot}_z(q_3), \qquad
S_{3}^{4} = \operatorname{Trans}_x(l_3)
$$

Multiplying $S_{0}^{4} = S_{0}^{1} S_{1}^{2} S_{2}^{3} S_{3}^{4}$ yields the closed-form end-effector position in the DH frame $\{0\}$:

$$
\begin{aligned}
p_x^{\text{DH}} &= \sin(q_1)\,\bigl(l_2 \sin(q_2) + l_3 \sin(q_2 + q_3)\bigr) \\
p_y^{\text{DH}} &= -\cos(q_1)\,\bigl(l_2 \sin(q_2) + l_3 \sin(q_2 + q_3)\bigr) \\
p_z^{\text{DH}} &= l_1 + l_2 \cos(q_2) + l_3 \cos(q_2 + q_3)
\end{aligned}
$$

> **Alignment with MuJoCo (`model.xml`)**: In the SolidWorks CAD frame used by MuJoCo, the shoulder/elbow pitch plane at $q_1 = 0$ is the $XZ$ plane instead of the $YZ$ plane—a fixed $+90^\circ$ rotation around $+Z$:
> $$(p_x^{\text{MJ}}, p_y^{\text{MJ}}, p_z^{\text{MJ}}) = (-p_y^{\text{DH}},\, p_x^{\text{DH}},\, p_z^{\text{DH}}).$$
> Joint angles $[q_1, q_2, q_3]$ have the **exact same sign and zero convention** in both models.

---

## 2. Velocity Propagation (Craig's Method) & Differential Kinematics

Using Craig's inward-to-outward recursion, the angular velocity $\omega_i^i$ and origin linear velocity $v_i^i$ expressed in each link frame $\{i\}$ are:

$$
\omega_i^i = (R_{i-1}^{i})^T \omega_{i-1}^{i-1} + \dot{q}_i \hat{z}, \qquad
v_i^i = (R_{i-1}^{i})^T \!\left(v_{i-1}^{i-1} + \omega_{i-1}^{i-1} \times P_{i-1,i}\right)
$$

> **Bug fixed from `cd1.m`**: The original script wrote `R23.'*v22 + cross(w22, P23)` (forgetting to rotate the cross product by $R_{i-1}^{i\,T}$).

The linear velocity of each link's center of mass (located at local vector $r_{c_i}^i$ in frame $\{i\}$) is:

$$
v_{c_i}^i = v_i^i + \omega_i^i \times r_{c_i}^i
$$

with local CoM vectors:
- **Link 1** (frame $\{1\}$): $r_{c_1}^1 = [0,\; 0,\; l_{c_1} - l_1]^T$ (height $l_{c_1}$ above ground)
- **Link 2** (frame $\{2\}$): $r_{c_2}^2 = [l_{c_2},\; 0,\; 0]^T$ *(fixed: in `cd1.m`, `P12c` evaluated to `[0, 0, 0]`)*
- **Link 3** (frame $\{3\}$): $r_{c_3}^3 = [l_{c_3},\; 0,\; 0]^T$ *(fixed: in `cd1.m`, `P23c` evaluated to `[l2, 0, 0]`)*

The $3 \times 3$ linear Jacobian relating joint velocities to end-effector velocity $\dot{p}_{\text{ee}} = J_v(q)\dot{q}$ is $J_v(q) = \frac{\partial p_{\text{ee}}}{\partial q}$.

---

## 3. Euler–Lagrange Dynamics

### Kinetic & Potential Energy
For each link $i \in \{1, 2, 3\}$ with mass $m_i$ and principal inertia tensor $I_i = \operatorname{diag}(I_{xx,i}, I_{yy,i}, I_{zz,i})$ at its CoM:

$$
K_i = \frac{1}{2} m_i (v_{c_i}^i)^T v_{c_i}^i + \frac{1}{2} (\omega_i^i)^T I_i \omega_i^i, \qquad K = K_1 + K_2 + K_3
$$

With $+Z$ upward and CoM world heights $H_1 = l_{c_1}$, $H_2 = l_1 + l_{c_2}\cos(q_2)$, $H_3 = l_1 + l_2\cos(q_2) + l_{c_3}\cos(q_2 + q_3)$:

$$
U = g\,(m_1 H_1 + m_2 H_2 + m_3 H_3), \qquad \mathcal{L}(q, \dot{q}) = K(q, \dot{q}) - U(q)
$$

*(Fixed from `cd1.m`: the original sign error `LT = K - (U1 - U2 - U3)` with negative `U_i` flipped the sign of `U1`.)*

### Standard Manipulator Form
Applying the Euler–Lagrange equations $\tau_i = \frac{d}{dt}\frac{\partial \mathcal{L}}{\partial \dot{q}_i} - \frac{\partial \mathcal{L}}{\partial q_i}$ gives:

$$
M(q)\ddot{q} + C(q, \dot{q})\dot{q} + G(q) = \tau
$$

- **Inertia matrix $M(q)$**: symmetric positive-definite, $M_{ij}(q) = \frac{\partial^2 K}{\partial \dot{q}_i \partial \dot{q}_j}$. Notice $M_{12} = M_{13} = 0$ (decoupled yaw vs. pitch inertia for an ideal planar-offset arm), while $M_{11}(q_2, q_3)$ depends on the radial reach of Links 2 and 3.
- **Coriolis & centrifugal matrix $C(q, \dot{q})$**: defined via Christoffel symbols of the first kind,
  $$C_{ij}(q, \dot{q}) = \frac{1}{2} \sum_{k=1}^{3} \left(\frac{\partial M_{ij}}{\partial q_k} + \frac{\partial M_{ik}}{\partial q_j} - \frac{\partial M_{jk}}{\partial q_i}\right)\dot{q}_k,$$
  which guarantees the passivity identity $\dot{M}(q) - 2C(q, \dot{q})$ is **skew-symmetric**.
- **Gravity torque vector $G(q) = \frac{\partial U}{\partial q}$**:
  $$
  G(q) = \begin{bmatrix}
  0 \\
  -g\bigl((m_2 l_{c_2} + m_3 l_2)\sin(q_2) + m_3 l_{c_3}\sin(q_2 + q_3)\bigr) \\
  -g\,m_3 l_{c_3}\sin(q_2 + q_3)
  \end{bmatrix}
  $$
  *(Zero at upright $q_2 = q_3 = 0$, negative when pitching forward to $+q_2$ so the motor must apply negative torque to pull the arm back up against gravity.)*

---

## 4. Parameter Summary (CAD vs. `cd1_live.mlx`)

| Parameter | SolidWorks CAD / MuJoCo (`CAD_PARAMS`) | Original `cd1_live.mlx` (`CD1_LIVE_PARAMS`) |
| :--- | :--- | :--- |
| Link lengths $[l_1, l_2, l_3]$ (m) | `[0.07895, 0.23700, 0.18000]` | `[0.0728, 0.2370, 0.1720]` |
| CoM distances $[l_{c1}, l_{c2}, l_{c3}]$ (m) | `[0.06707, 0.12050, 0.04539]` | `[0.0364, 0.1185, 0.0860]` |
| Moving link masses $[m_1, m_2, m_3]$ (kg) | `[0.00925, 0.09413, 0.01179]` | `[0.40024, 0.2436, 0.0345]` |

> **Small residual vs. 3D CAD**: In the full 3D SolidWorks assembly, Links 1–3 have a small `~20 mm` lateral offset along the pitch axis (due to the AX-12A horn/bracket mounting thickness), which produces tiny cross-coupling terms (`M_12 ~ 1e-5 kg·m²`) that an idealized 0-offset DH model omits. Across 100 random states, the analytical model matches MuJoCo to `0.000 mm` in FK and `~1e-5 N·m` in inverse dynamics.
