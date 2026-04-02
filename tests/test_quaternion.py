# norm, multiply, conjugate, kinematics
import numpy as np
from core.quaternion import quat_normalize, rotation_matrix

def test_quaternion():
    q = np.array([2.0, 0.0, 0.0, 0.0])
    qn = quat_normalize(q)
    assert np.isclose(np.linalg.norm(qn), 1.0)

def test_quaternion_matrix_orthoganality():
    q = quat_normalize(np.random.rand(4))
    R = rotation_matrix(q)
    # R * R^T = I
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)

    # det(R) = 1 (proper rotaion and not a reflection)
    assert np.isclose(np.linalg.det(R), 1.0, atol = 1e-10)