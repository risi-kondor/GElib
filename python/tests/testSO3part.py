import torch
import gelib as G
import pytest

class TestSO3part(object):
    
    @pytest.mark.parametrize('b', [1, 2, 4])    
    @pytest.mark.parametrize('l', [1, 2, 4, 8])
    @pytest.mark.parametrize('n', [1, 2, 4, 8, 32])
    def test_CGproduct(self,b,l,n):
        x = G.SO3part.randn(b,l,n)
        y = G.SO3part.randn(b,l,n)
        z=G.CGproduct(x,y,l)

        R = G.SO3element.uniform()
        xr=x.rotate(R)
        yr=y.rotate(R)
        zr=G.CGproduct(xr,yr,l)
        rz=z.rotate(R)

        torch.allclose(rz,zr,rtol=1e-3, atol=1e-4)


    @pytest.mark.parametrize('b', [1, 2, 4])    
    @pytest.mark.parametrize('l', [1, 2, 4, 8])
    @pytest.mark.parametrize('n', [1, 2, 4, 8, 32])
    def test_DiagCGproduct(self,b,l,n):
        x=G.SO3part.randn(b,l,n)
        y=G.SO3part.randn(b,l,n)
        z=G.DiagCGproduct(x,y,l)

        R=G.SO3element.uniform()
        xr=x.rotate(R)
        yr=y.rotate(R)
        zr=G.DiagCGproduct(xr,yr,l)
        rz=z.rotate(R)

        assert torch.allclose(rz,zr,rtol=1e-3, atol=1e-5)


    @pytest.mark.parametrize('b', [1, 2, 4])    
    @pytest.mark.parametrize('l', [1, 2, 4, 8])
    @pytest.mark.parametrize('n', [1, 2, 4, 8, 32])
    def test_CGproduct_backprop(self,b,l,n):
        x = G.SO3part.randn(b,l,n)
        y = G.SO3part.randn(b,l,n)
        x.requires_grad_()
        y.requires_grad_()
        torch.autograd.gradcheck(G.SO3part_CGproductFn.apply,[x,y,l])

