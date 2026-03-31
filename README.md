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
l_tvc = Distance from CoM to engine gimbal point  [m] 
rho(h) = Atmospheric density at altitude h  [kg/m^3] 
CD = Drag coefficient  [-] 
Aref = Reference area = pi*D^2/4  [m^2] 
R(q) = 3x3 rotation matrix from quaternion (body→inertial) 
[v]x = Skew-symmetric (cross-product) matrix of vector v 
x_dot = Time derivative dx/dt
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