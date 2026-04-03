NOTATION:
r = Position vector in inertial (flat-Earth) frame  [m] 
v = Velocity vector in inertial frame  [m/s] 
q = Unit quaternion [q0, q1, q2, q3]^T  (scalar first, body→inertial) 
omega = Angular velocity in body wrt inertial, expressed in body frame  [rad/s]  = [wx, wy, wz]^T 
m = Rocket mass  [kg] 
T = Thrust magnitude  [N] 
Isp = Specific impulse  [s] 
g0 = Standard gravity  9.80665 m/s^2 
I_body = Inertia tensor (diagonal for symmetric rocket)  [kg·m^2] 
delta_p/y = TVC gimbal angle: pitch / yaw  [rad] 
r_c2tvc = Distance from CoM to engine gimbal point  [m] 
rho(h) = Atmospheric density at altitude h  [kg/m^3] 
CD = Drag coefficient  [-] 
Aref = Reference area = pi*D^2/4  [m^2] 
R(q) = 3x3 rotation matrix from quaternion (body→inertial) 
[v]x = Skew-symmetric (cross-product) matrix of vector v 
x_dot = Time derivative dx/dt
-------------------------------------------------------------------------
TO RUN TESTS: python -m pytest -v
-------------------------------------------------------------------------
STATE VECTORS:
x = [ r_x, r_y, r_z,          <!--  position in inertial frame  (3)  -->
v_x, v_y, v_z,          <!-- # velocity in inertial frame  (3)  -->
q0, q1, q2, q3,         <!-- # unit quaternion body→inertial (4)  -->
wx, wy, wz,           <!-- # angular velocity in body frame (3)  -->
m   <!-- # total mass               (1)-->
]                   
<!-- # total: 13 states  -->
u = [ delta_p, delta_y ]      <!-- # TVC gimbal angles  [rad]  (2)  -->
-------------------------------------------------------------------------
REFERENCE FRAMES:
<!-- NED -->
Origin at launch pad. X points East, Y points North, Z points Up. This is fixed — it does not rotate 
with the Earth. Acceptable for short burns (< 5 min). Gravity points in the -Z direction.

<!-- BDY -->
Origin at the rocket's center of mass (CoM). Z_body points along the thrust axis (nose up in neutral 
attitude). X_body and Y_body complete a right-handed set. The TVC gimbal deflects the thrust 
vector relative to the body Z axis.


                     ^ z(BDY)   
                     |   
                     |   
                    /\               O => out of the plane, 
                   /  \                   rotaion is positive
                  |    |
                  |    |
                  |    |
                  |    |
                  |    |
                  |    |
                  |    |
  y(BDY)    O     | 0  |----------> x(BDY)
                  |    |
                  |    |
                  |    |
                  |    |
                  |    |
                  |    |
                    /\   
                   /  \   
<!-- The quaternion q encodes the rotation that takes a vector from body frame to inertial frame:  -->
v_NED = R(q) * V_BDY

=========================IMPORTANT CONSIDERATIONS=====================
Ground Contact, Liftoff, and Thrust Build-Up — Problem Statement

In the current 6-DOF rocket simulation, the vehicle is modeled using unconstrained rigid-body dynamics:

𝑣_dot = 1/m (F_th + F_g + F_aero)
This formulation assumes free-flight from t=0 and does not account for ground interaction or engine startup transients, leading to physically incorrect behavior during the launch phase.

Key Issues to Address
1. Ground Contact Constraint

The current model implicitly assumes a flat ground at 
z=0. However:
Real terrain is non-uniform (mountains, valleys, sea level variations)
Ground should be modeled as a surface function, not a constant plane:
𝑧 ≥ ℎ_terrain(x, y)
Where:
h_terrain(x,y) is the terrain elevation at horizontal position

Problem:
Without enforcing this constraint, the rocket may:
Penetrate the ground numerically
Exhibit non-physical motion before liftoff

2. Liftoff Condition (Thrust vs Weight)

Liftoff does not occur when the engine turns on, but only when:

𝐹_thrust, z > mg
Where:
F_thrust,z is the vertical component of thrust in the inertial frame

Problem:
If 
F_thrust,z ≤ mg, the rocket must remain stationary, enforced via a ground reaction force or constraint.

3. Thrust Build-Up (Engine Transient)
Real engines do not produce full thrust instantly:
T(t)→T_max  over a short time interval

Typical behavior:
Ignition phase: T(t)<mg
Ramp phase: T(t)↑
Liftoff occurs only after threshold crossing

Problem:
Current model assumes:
Instantaneous T=T_max
This ignores the finite ignition transient, causing unrealistic early motion

Core Modeling Requirement
A physically consistent launch model must satisfy:
Terrain-aware ground constraint
z≥h_terrain(x,y)
Conditional motion
If F_thrust,z ≤ mg: rocket remains on ground
If F_thrust,z > mg: rocket transitions to free-flight

Time-varying thrust
T=T(t)

Open Design Question
How should the simulation architecture incorporate:
Ground contact constraints (position + velocity)
Terrain modeling h_terrain(x,y)
Thrust ramp dynamics
Liftoff detection
…while maintaining numerical stability and modularity within the ODE + RK4 framework?

Scope Note
For V1 (flat-Earth approximation), terrain may be simplified as:
h_terrain(x,y)=0
But the formulation should remain extensible to:
Digital elevation maps (DEM)
Non-flat launch sites
Ocean launches (dynamic surface)