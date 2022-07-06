import torch
import gelib as G
import pytest

class TestSO3partArr(object):
    
    @pytest.mark.parametrize('a', [1, 2, 4])    
    @pytest.mark.parametrize('l', [1, 2, 4, 8])
    @pytest.mark.parametrize('n', [1, 2, 4, 8, 32])
    def test_CGproduct(self,a,l,n):
        x = G.SO3partArr.randn([a,a],l,n)
        y = G.SO3partArr.randn([a,a],l,n)
        z=G.CGproduct(x,y,l)

        R = G.SO3element.uniform()
        xr=x.rotate(R)
        yr=y.rotate(R)
        zr=G.CGproduct(xr,yr,l)
        rz=z.rotate(R)

        torch.allclose(rz,zr,rtol=1e-3, atol=1e-4)


    @pytest.mark.parametrize('a', [1, 2, 4])    
    @pytest.mark.parametrize('l', [1, 2, 4, 8])
    @pytest.mark.parametrize('n', [1, 2, 4, 8, 32])
    def test_DiagCGproduct(self,a,l,n):
        x=G.SO3partArr.randn([a,a],l,n)
        y=G.SO3partArr.randn([a,a],l,n)
        z=G.DiagCGproduct(x,y,l)

        R=G.SO3element.uniform()
        xr=x.rotate(R)
        yr=y.rotate(R)
        zr=G.DiagCGproduct(xr,yr,l)
        rz=z.rotate(R)

        assert torch.allclose(rz,zr,rtol=1e-3, atol=1e-5)


    @pytest.mark.parametrize('a', [1, 2, 4])    
    @pytest.mark.parametrize('l', [1, 2, 4, 8])
    @pytest.mark.parametrize('n', [1, 2, 4, 8, 32])
    def test_CGproduct_backprop(self,a,l,n):
        return
        x = G.SO3partArr.randn([a,a],l,n)
        y = G.SO3partArr.randn([a,a],l,n)
        x.requires_grad_()
        y.requires_grad_()
        torch.autograd.gradcheck(G.SO3partArr_CGproductFn.apply,[x,y,l])

