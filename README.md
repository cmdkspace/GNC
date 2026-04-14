# GNC — Guidance, Navigation & Control Simulation Framework

A modular, research-grade 6-DOF rocket simulation and control framework written in Python. Designed for studying ascent dynamics, attitude stabilization, and trajectory optimization under realistic aerodynamic and propulsion models.

> **Status:** V1 Complete — Flat-Earth 6-DOF with TVC attitude control and offline trajectory optimization.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Physics & Mathematical Foundations](#physics--mathematical-foundations)
   - [Reference Frames](#reference-frames)
   - [State Vector](#state-vector)
   - [Equations of Motion](#equations-of-motion)
   - [Forces](#forces)
   - [Torques](#torques)
   - [Quaternion Kinematics](#quaternion-kinematics)
   - [Rotational Dynamics (Euler)](#rotational-dynamics-euler)
   - [Mass Depletion (Tsiolkovsky)](#mass-depletion-tsiolkovsky)
4. [Modules](#modules)
   - [core/](#core)
   - [control/](#control)
   - [optimization/](#optimization)
   - [analysis/](#analysis)
   - [config/](#config)
   - [scripts/](#scripts)
5. [Simulation Modes](#simulation-modes)
6. [Vehicle Parameters (V1)](#vehicle-parameters-v1)
7. [Running the Simulation](#running-the-simulation)
8. [Testing](#testing)
9. [Known Behaviors & Design Notes](#known-behaviors--design-notes)
10. [V1 Assumptions & Limitations](#v1-assumptions--limitations)
11. [Versioning Roadmap](#versioning-roadmap)
12. [Dependencies](#dependencies)

---

## Overview

This framework simulates a single-stage rocket undergoing powered vertical ascent under a flat-Earth approximation. The simulation solves the full 6-DOF (six degrees of freedom) rigid-body equations of motion and supports three distinct operating modes:

| Mode | Description |
|---|---|
| **Baseline** | Open-loop, zero-control ascent. Pure physics. |
| **PD/LQR Controlled** | Closed-loop attitude stabilization via quaternion-error PD/LQR + TVC. |
| **Optimized** | Offline trajectory optimization over pitch program and thrust cutoff using L-BFGS-B. |

All three modes share the same ODE core (`rocket_ode`) and integrator (`rk4`), ensuring physics consistency across comparisons.

---

## Project Structure

```
GNC/
├── main.py                    # Entry point: runs all 3 cases and plots comparison
│
├── core/
│   ├── atmosphere.py          # US Std Atmosphere (exponential), gravity model
│   ├── quaternion.py          # Quaternion math: rotation matrix, kinematics, error
│   ├── rocket_model.py        # Master ODE: thrust, drag, TVC torque, Euler equations
│   └── integrator.py          # RK4 integrator + simulation loop with burnout logic
│
├── control/
│   ├── pd_controller.py       # Quaternion-error PD controller → gimbal commands
│   └── lqr.py                 # (Stub) LQR controller — planned for V2 but implemented partially in V1 as well
│
├── optimization/
│   ├── cost_function.py       # Cost J(θ), pitch profile, control law builder
│   └── optimizer.py           # L-BFGS-B wrapper whihc is essentially "Quasi - Linerizzation Method" (scipy.optimize.minimize) 
│
├── analysis/
│   └── plots.py               # Altitude comparison, pitch, control input plots
│
├── config/
│   ├── vehicle.yaml           # Vehicle physical parameters (mass, inertia, aero, limits)
│   └── sim_congig.yaml        # Simulation time/step parameters (stub)
│
├── scripts/
│   ├── run_baseline.py        # Standalone baseline ascent script with detailed plots
│   ├── run_controlled.py      # Standalone PD-controlled ascent with full diagnostic output
│   └── run_optimized.py       # Standalone optimized trajectory script
│
├── tests/
│   ├── test_quaternion.py     # Quaternion normalization, rotation matrix orthogonality
│   ├── test_forces.py         # Gravity direction/magnitude, atmosphere decay, drag laws
│   ├── test_integrator.py     # RK4 kinematic accuracy, quaternion norm preservation
│   ├── test_rocket_ode.py     # ODE structural correctness (7 unit tests)
│   ├── test_physics.py        # Tsiolkovsky delta-V validation, mass depletion rate
│   └── test_controller.py     # PD controller: zero-error, pitch/yaw decoupling, saturation
│
└── requirements.txt           # numpy, scipy, matplotlib, pyyaml, pytest
```

---

## Physics & Mathematical Foundations

### Reference Frames

Two coordinate frames are used throughout:

#### NED (North-East-Down, used as Inertial Frame)

- **Origin:** Launch pad (fixed)
- **Axes:** X → East, Y → North, Z → Up
- Treated as inertial for short burns (< ~5 min). Earth rotation neglected.
- Gravity acts in the −Z direction: `F_g = [0, 0, −mg]ᵀ`

#### BDY (Body Frame)

- **Origin:** Rocket center of mass (CoM)
- **Z_body:** Along the thrust axis (nose up in neutral attitude)
- X_body and Y_body form a right-handed set as in z = y * x 
- TVC gimbal deflects the thrust vector relative to Z_body

```
                ^ z(BDY)
                |
                |
               /\               O => out of the plane
              /  \                   rotation is positive
             |    |
             |    |
             |    |
             |    |
             |    |
y(BDY)  O    | 0  |----------> x(BDY)
             |    |
             |    |
               \/
              /  \
```

The quaternion `q` encodes the rotation from body frame to inertial frame:

```
v_NED = R(q) · v_BDY
```

---

### State Vector

The simulation propagates a 14-dimensional state vector:

```
x = [ r_x, r_y, r_z,       ← position in inertial frame  [m]      (indices 0–2)
      v_x, v_y, v_z,       ← velocity in inertial frame  [m/s]    (indices 3–5)
      q0, q1, q2, q3,      ← unit quaternion body→inertial [-]     (indices 6–9)
      wx, wy, wz,          ← angular velocity in body frame [rad/s](indices 10–12)
      m   ]                ← total mass  [kg]                       (index 13)
```

Control input (2-dimensional):

```
u = [ δ_p, δ_y ]           ← TVC gimbal angles: pitch / yaw  [rad]
```

---

### Equations of Motion

The full ODE system integrated at each time step:

```
ṙ      = v
v̇      = (F_thrust + F_gravity + F_aero) / m
q̇      = ½ Ξ(q) · ω
ω̇      = I⁻¹ · (τ_TVC − ω × (I · ω))
ṁ      = −T / (Isp · g₀)
```

---

### Forces

#### Thrust (Body Frame → NED)

The thrust vector is produced by a gimbal-deflected engine. In the body frame:

```
F_thrust,body = T · [ −sin(δ_y),  sin(δ_p),  cos(δ_p)·cos(δ_y) ]ᵀ
```

Rotated to NED via the rotation matrix `R(q)`:

```
F_thrust,NED = R(q) · F_thrust,body
```

Gimbal limit enforced: `|δ| ≤ 5° = 0.0873 rad`

#### Gravity

Constant gravitational field (flat-Earth):

```
F_gravity = [0, 0, −m·g₀]ᵀ    where g₀ = 9.80665 m/s²
```

#### Aerodynamic Drag

Exponential US Standard Atmosphere model:

```
ρ(h) = ρ₀ · exp(−h / Hₛ)      ρ₀ = 1.225 kg/m³,  Hₛ = 8500 m
```

Drag force (returned as NED vector, opposes velocity):

```
F_aero = −½ · ρ(h) · CD · Aref · |v| · v
```

Zero drag is enforced when `|v|² < 1e-10` to avoid numerical singularities.

---

### Torques

#### TVC Torque

The only active torque in V1. The engine gimbal point is located at `−r_c2tvc` along the body Z axis from CoM:

```
r_engine = [0, 0, −r_c2tvc]ᵀ   (body frame)
τ_TVC    = r_engine × F_thrust,body
```

> Aerodynamic torques are deferred to V2. For V1, the center of pressure coincides with CoM by assumption.

---

### Quaternion Kinematics

The unit quaternion `q = [q₀, q₁, q₂, q₃]ᵀ` (scalar-first convention) evolves as:

```
q̇ = ½ · [ −wx·q₁ − wy·q₂ − wz·q₃ ]
         [  wx·q₀ + wz·q₂ − wy·q₃ ]
         [  wy·q₀ − wz·q₁ + wx·q₃ ]
         [  wz·q₀ + wy·q₁ − wx·q₂ ]
```

**Drift prevention:** The quaternion is re-normalized after every RK4 step:

```python
X_new[6:10] = q / ||q||
```

**Attitude error** for the PD controller is computed as:

```
q_e = q_ref⁻¹ ⊗ q
```

The shortest rotation path is enforced by negating `q_e` if `q_e[0] < 0`.

**Rotation matrix** (body → inertial):

```
R(q) = [ 1−2(q₂²+q₃²)    2(q₁q₂−q₀q₃)    2(q₁q₃+q₀q₂) ]
        [ 2(q₁q₂+q₀q₃)    1−2(q₁²+q₃²)    2(q₂q₃−q₀q₁) ]
        [ 2(q₁q₃−q₀q₂)    2(q₂q₃+q₀q₁)    1−2(q₁²+q₂²) ]
```

---

### Rotational Dynamics (Euler)

Full Euler equations for a rigid body with diagonal inertia tensor:

```
ẇx = ((Iyy − Izz) · wy · wz + τx) / Ixx
ẇy = ((Izz − Ixx) · wz · wx + τy) / Iyy
ẇz = ((Ixx − Iyy) · wx · wy + τz) / Izz
```

The inertia tensor is assumed diagonal (axially symmetric rocket):
`I_body = (Ixx, Iyy, Izz)`.

---

### Mass Depletion (Tsiolkovsky)

Propellant burn rate derived from rocket equation:

```
ṁ = −T / (Isp · g₀)
```

Thrust cutoff conditions (both checked in the simulation loop):
1. **Mass-based:** When `m ≤ m_dry`, thrust is cut and mass is clamped.
2. **Time-based:** When `t > t_cutoff` (set by optimizer), thrust is zeroed.

---

## Modules

### `core/`

| File | Purpose |
|---|---|
| `atmosphere.py` | `air_density(h)` — exponential atmosphere. `gravity_force(m)` — NED gravity vector. |
| `quaternion.py` | `rotation_matrix(q)`, `quat_multiply`, `quat_conjugate`, `quat_normalize`, `quat_kinematics(q, ω)`, `quaternion_error(q, q_ref)` |
| `rocket_model.py` | `rocket_ode(T, X, u, params)` — master ODE computing `Ẋ`. Includes `thrust_body`, `thrust_NED`, `aero_drag`, `tvc_torque`, `rot_dynamics_euler`, `mass_rate`. |
| `integrator.py` | `rk4(f, t, X, dt, u, params)` — 4th-order Runge-Kutta step with quaternion re-normalization. `simulate(X0, u_func, t_span, dt, params)` — full simulation loop with early termination on ground impact. |

### `control/`

| File | Purpose |
|---|---|
| `pd_controller.py` | `pd_controller(t, x, q_ref, gains)` — Computes quaternion-error PD control, returns `[δ_p, δ_y]` clipped to `delta_max`. |
| `lqr.py` | **Stub.** LQR controller placeholder for V2. |

**PD Control Law:**

```
δ_p = −(Kp · q_e[1] + Kd · ωx)   (pitch channel)
δ_y = −(Kp · q_e[2] + Kd · ωy)   (yaw channel)
```

Where `q_e[1:4]` is the vector part of the quaternion attitude error.

### `optimization/`

| File | Purpose |
|---|---|
| `cost_function.py` | `cost_function(θ, params, sim_params, gains)` — evaluates total cost J. Also provides `pitch_profile(t, θ)`, `control_law_build(gains, θ)`, `initial_state_build(θ, params)`. |
| `optimizer.py` | `run_optimizer(θ₀, bounds, params, sim_params, gains, q_ref)` — wraps `scipy.optimize.minimize` with L-BFGS-B. |

**Optimization Variable Vector θ:**

| Index | Parameter | Bounds | Description |
|---|---|---|---|
| `θ[0]` | `θ_max` | (−0.08, 0.08) rad | Maximum pitch command angle |
| `θ[1]` | `t_cutoff` | (1.0, burn_time) s | Engine cutoff time |
| `θ[2]` | `t_turn` | (0.5, 5.0) s | Time to begin pitch-over maneuver |
| `θ[3]` | `t_ramp` | (0.5, 5.0) s | Duration of pitch ramp-up |

**Pitch Profile (linear ramp):**

```
pitch(t) = 0                                      if t < t_turn
         = θ_max · (t − t_turn) / t_ramp          if t_turn ≤ t < t_turn + t_ramp
         = θ_max                                   if t ≥ t_turn + t_ramp
```

**Cost Function:**

```
J(θ) = w₁ · (h_target − h_max)² + w₂ · ∫(δ_p² + δ_y²) dt
```

| Parameter | Default Value | Role |
|---|---|---|
| `h_target` | 5000 m | Target apogee altitude |
| `w₁` | 10.0 | Altitude tracking weight |
| `w₂` | 0.01 | Control effort penalty weight |

### `analysis/`

| Function | Description |
|---|---|
| `plot_altComp(...)` | Overlaid altitude vs. time for all 3 simulation modes |
| `plot_pitch(t, X)` | Pitch angle (from quaternion) vs. time |
| `plot_control(t, u)` | TVC gimbal angles `δ_p` and `δ_y` vs. time |

### `config/`

| File | Contents |
|---|---|
| `vehicle.yaml` | All vehicle physical parameters: mass, propulsion, geometry, inertia, aerodynamics, control limits, PD gains |
| `sim_congig.yaml` | Simulation time/step configuration (stub, not yet loaded programmatically) |

### `scripts/`

Standalone runnable scripts for individual simulation modes with extended diagnostic output and plots:

| Script | Description |
|---|---|
| `run_baseline.py` | Open-loop vertical ascent, plots altitude / velocity / mass vs. time |
| `run_controlled.py` | PD-stabilized ascent with initial pitch perturbation (`q₁=0.05`), detailed attitude and control history plots |
| `run_optimized.py` | Full offline optimization + re-simulation of optimal trajectory |

---

## Simulation Modes

### Baseline (Open-Loop)

No control input (`u = [0, 0]` always). The rocket ascends vertically under thrust, gravity, and drag. Useful for validating raw physics.

```bash
python scripts/run_baseline.py
```

### PD Controlled

A small initial pitch disturbance (`q₁ = 0.05 rad`) is applied. The PD controller actively drives the quaternion error to zero via TVC. Demonstrates attitude stabilization and damping.

```bash
python scripts/run_controlled.py
```

A known minor behavior: After burnout when `T = 0`, TVC produces no torque and the system enters free-body Euler dynamics. A very small attitude drift (`|q₁| ~ 2e-5`, i.e., ~0.001°) accumulates from RK4 floating-point residuals and quaternion renormalization bias. This is **physically expected** and numerically insignificant — real rockets behave the same without RCS or aerodynamic stabilization post-burnout.

### Optimized

Runs L-BFGS-B optimization over `θ = [θ_max, t_cutoff, t_turn, t_ramp]` to minimize the cost function `J(θ)`, then re-simulates the optimal trajectory.

```bash
python scripts/run_optimized.py
```

### All Three (Comparison)

Runs all three modes and produces a single overlaid altitude comparison plot.

```bash
python main.py
```

---

## Vehicle Parameters (V1)

Defined in `config/vehicle.yaml` and mirrored in `main.py` / scripts:

| Parameter | Symbol | Value | Units |
|---|---|---|---|
| Initial mass | m₀ | 100.0 | kg |
| Dry mass | m_dry | 50.0 | kg |
| Max thrust | T_max | 15,000 | N |
| Specific impulse | Isp | 260 | s |
| Body diameter | D | 0.5 | m |
| Reference area | Aref | π(D/2)² ≈ 0.196 | m² |
| Drag coefficient | CD | 0.4 | — |
| CoM-to-gimbal distance | r_c2tvc | 2.0 | m |
| Pitch moment of inertia | Ixx | 2000 | kg·m² |
| Yaw moment of inertia | Iyy | 2000 | kg·m² |
| Roll moment of inertia | Izz | 200 | kg·m² |
| Max gimbal angle | δ_max | 0.0873 (5°) | rad |
| PD proportional gain | Kp | 0.6 | — |
| PD derivative gain | Kd | 0.3 | — |
| Simulation duration | tf | 60 | s |
| Integration timestep | dt | 0.01 | s |

**Theoretical burn time:**

```
t_burn = (m₀ − m_dry) / (T_max / (Isp · g₀)) ≈ 8.5 s
```

---

## Running the Simulation

### Prerequisites

```bash
pip install numpy scipy matplotlib pyyaml pytest
```

### Run all three cases with comparison plot

```bash
python main.py
```

### Individual scripts

```bash
# Baseline (open-loop)
python scripts/run_baseline.py

# PD attitude control
python scripts/run_controlled.py

# Trajectory optimization
python scripts/run_optimized.py
```

> **Note:** All scripts must be run from the project root so that relative imports (e.g., `from core.integrator import simulate`) resolve correctly.

---

## Testing

The test suite covers physics validation, numerical accuracy, and controller correctness. Run with:

```bash
python -m pytest -v
```

### Test Coverage Summary

| Test File | Tests | What is Verified |
|---|---|---|
| `test_quaternion.py` | 2 | Normalization correctness; Rotation matrix orthogonality and det(R)=1 |
| `test_forces.py` | 5 | Gravity direction/magnitude; Atmosphere monotonic decay + sea-level density; Drag opposes velocity; Drag = 0 at rest; Drag ∝ v² |
| `test_integrator.py` | 3 | Constant-accel kinematics (z = ½gt²); Velocity growth under gravity (v = gt); Quaternion unit-norm preservation after integration |
| `test_rocket_ode.py` | 7 | ṙ = v identity; Gravity-only acceleration; Drag-opposing velocity direction; Drag quadratic scaling; Mass decreases with thrust; No mass change without thrust; Zero TVC → zero torque; Zero ω → zero q̇ |
| `test_physics.py` | 2 | Tsiolkovsky Δv within 5% (with gravity loss correction); Mass depletion monotonic and rate-accurate to 1% |
| `test_controller.py` | 6 | Quaternion error identity; Zero error → zero command; Pitch-error → pitch command only; Yaw-error → yaw command only; Damping opposes angular velocity; Saturation clamps to `delta_max` |

**Total: 25 unit tests** across physics, mathematics, integration, and control.

---

## Known Behaviors & Design Notes

### Post-Burnout Attitude Drift

After engine cutoff, `T = 0` means `τ_TVC = 0`. The system enters free-body Euler dynamics with no restoring or damping torque. Tiny residual angular velocities from RK4 floating-point and quaternion renormalization bias integrate into a slow attitude drift (`|q₁| ~ 2e-5`, ~0.001°). This is **physically correct** behavior.

### No Ground Constraint in V1

The current model uses unconstrained free-flight dynamics from `t = 0`. There is no liftoff condition, no terrain model, and no thrust-buildup transient. The simulation terminates early if `z < 0` after `t > 1 s`. Ground contact, liftoff physics, and engine startup transients are scheduled for V3.

### Atmospheric Model

A single-layer exponential fit to the US Standard Atmosphere is used:
`ρ(h) = 1.225 · exp(−h/8500)` kg/m³. This gives ~3% error vs. ISA below 100 km altitude. A multi-layer ISA model is planned for V2.

### CD Mach Independence

The drag coefficient `CD = 0.4` is treated as constant (Mach-independent). Compressibility effects and transonic drag rise are deferred to a later version.

### Aerodynamic Torques

TVC is the sole torque source in V1. Aerodynamic torques (which require a separate center of pressure model) are deferred to V2.

---

## V1 Assumptions & Limitations

| Assumption | Impact | Future Version |
|---|---|---|
| Flat-Earth, non-rotating frame | Valid for burns < ~5 min; ignores Coriolis | V4/V5 |
| Point mass ground (z = 0 plane) | No launch rail, no liftoff physics | V3 |
| Instantaneous full thrust at t = 0 | Ignores engine startup transient | V3 |
| Constant Isp | Ignores throttle / mixture ratio variation | V3 |
| Constant CD (Mach-independent) | Ignores transonic drag rise | V2 |
| Exponential atmosphere only | Less accurate above 20 km | V2 |
| No aerodynamic torque | CoP = CoM assumed | V2 |
| Diagonal inertia tensor | Symmetric rocket, no products of inertia | V2 |
| LQR stub (not implemented) | Currently a placeholder file | V2 |
| Config YAML not loaded programmatically | Parameters hardcoded in scripts | V2 |

---

## Versioning Roadmap

| Version | Theme | Key Features |
|---|---|---|
| **V1** ✅ | **6-DOF Baseline + Control + Optimization** | Flat-Earth rigid-body ODE, RK4, TVC PD attitude control, L-BFGS-B pitch-program optimization, 25 unit tests |
| **V2** | **Aerodynamics & Control Enhancement** | Multi-layer ISA atmosphere, Mach-dependent CD, aerodynamic torque with CoP model, full YAML config loading, LQR controller implementation |
| **V3** | **Realistic Launch Physics** | Ground contact constraint, terrain model h(x,y), engine startup transient (thrust buildup), liftoff detection (F_thrust,z > mg), RCS stub |
| **V4** | **Navigation & Guidance** | IMU noise model, EKF state estimator, gravity turn guidance, pitch program scheduler |
| **V5** | **Global Frame & Precision** | Rotating Earth (Coriolis), WGS84 ellipsoid, high-fidelity atmosphere, Monte Carlo dispersion analysis |

---

## Dependencies

```
numpy        # State vector operations, linear algebra
scipy        # L-BFGS-B optimizer (scipy.optimize.minimize), numerical integration utilities
matplotlib   # All trajectory and diagnostic plots
pyyaml       # Vehicle and simulation configuration files
pytest       # Unit test framework
```

Install all dependencies:

```bash
pip install numpy scipy matplotlib pyyaml pytest
```

---

## Notation Reference

| Symbol | Description | Units |
|---|---|---|
| `r` | Position vector in inertial frame | m |
| `v` | Velocity vector in inertial frame | m/s |
| `q = [q₀,q₁,q₂,q₃]ᵀ` | Unit quaternion (scalar-first, body→inertial) | — |
| `ω = [wx,wy,wz]ᵀ` | Angular velocity in body frame | rad/s |
| `m` | Rocket total mass | kg |
| `T` | Thrust magnitude | N |
| `Isp` | Specific impulse | s |
| `g₀` | Standard gravity (9.80665) | m/s² |
| `I_body` | Diagonal inertia tensor | kg·m² |
| `δ_p / δ_y` | TVC gimbal angles: pitch / yaw | rad |
| `r_c2tvc` | Distance from CoM to engine gimbal point | m |
| `ρ(h)` | Atmospheric density at altitude h | kg/m³ |
| `CD` | Drag coefficient | — |
| `Aref` | Reference area = π·D²/4 | m² |
| `R(q)` | 3×3 rotation matrix (body→inertial) | — |
| `[v]×` | Skew-symmetric (cross-product) matrix of v | — |
| `ẋ` | Time derivative dx/dt | — |