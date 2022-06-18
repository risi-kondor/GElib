import torch
import numpy as np
import gelib
import pytest
from scipy.spatial.transform import Rotation


class TestSpharmEquivariance():
    @pytest.mark.parametrize('ell', [0, 1, 4, 10])
    @pytest.mark.parametrize('num_batches', [1, 5, 64])
    def test_part_spharm(self, ell, num_batches):
        X = torch.randn(num_batches, 3)

        euler_angles = np.random.randn(3)
        print(euler_angles)
        # rot = Rotation.from_euler("ZYZ", angles=euler_angles)
        rot = Rotation.from_euler("XYZ", angles=euler_angles)
        R = gelib.SO3element(euler_angles[0], euler_angles[1], euler_angles[2])
        rot_mat = rot.as_matrix()

        X_rot = X @ torch.from_numpy(rot_mat).float().t()

        y = gelib.SO3part.spharmB(ell, X)
        y_rot = gelib.SO3part.spharmB(ell, X_rot)

        assert(torch.allclose(y_rot, y.rotate(R)))



