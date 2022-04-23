import torch
import gelib as G
import pytest

class TestCGlayer(object):
    @pytest.mark.parametrize('tau', [1, 2, 4, 8, 16, 32, 64])
    @pytest.mark.parametrize('maxl', range(7))
    @pytest.mark.parametrize('batch_size', [1, 2, 4])
    def test_equivariance(self, batch_size,tau, maxl):
        x = G.SO3vec.randn(batch_size, [tau for i in range(maxl + 1)])
        y = G.SO3vec.randn(batch_size, [tau for i in range(maxl + 1)])
        x_rot = G.SO3vec.zeros_like(x)
        y_rot = G.SO3vec.zeros_like(y)
        R = G.SO3element.uniform()
        for i in range(maxl + 1):
            x_rot.parts[i] =x.parts[i].detach().apply(R)
        for i in range(maxl + 1):
            y_rot.parts[i] =y.parts[i].detach().apply(R)


        x_out = G.DiagCGproduct(x, y, maxl=maxl)
        x_rot_out = G.DiagCGproduct(x_rot, y_rot, maxl=maxl)

        x_out_rot = G.SO3vec.zeros_like(x_out)
        for i in range(maxl+1 ):
            x_out_rot.parts[i] = x_out.parts[i].detach().apply(R)
            assert (torch.allclose(x_out_rot.parts[i] , x_rot_out.parts[i], rtol=1e-4, atol=1e-5))
