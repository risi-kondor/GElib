import torch
import gelib as G
import pytest

class TestSO3partArr(object):
    
    def partArr_partArr_backprop(self,b,adims,l,n,fn,arg0):
        x = G.SO3partArr.randn(b,adims,l,n)
        y = G.SO3partArr.randn(b,adims,l,n)
        x.requires_grad_()
        y.requires_grad_()
        z=fn(x,y,arg0)

        test_vec=G.SO3partArr.randn_like(z)
        loss=z.odot(test_vec)
        loss.backward(torch.tensor(1.0))
        xgrad=x.grad
        ygrad=y.grad

        xeps=G.SO3partArr.randn_like(x)
        z=fn(x+xeps,y,arg0)
        xloss=z.odot(test_vec)
        assert(torch.allclose(xloss-loss,xeps.odot(xgrad),rtol=1e-3, atol=1e-4))

        yeps=G.SO3partArr.randn_like(x)
        z=fn(x,y+yeps,arg0)
        yloss=z.odot(test_vec)
        assert(torch.allclose(yloss-loss,yeps.odot(ygrad),rtol=1e-3, atol=1e-4))


    @pytest.mark.parametrize('b', [1, 2, 4])    
    @pytest.mark.parametrize('a', [1, 2, 4])    
    @pytest.mark.parametrize('l', [1, 2, 4, 8])
    @pytest.mark.parametrize('n', [1, 2, 4, 8, 32])
    def test_CGproduct(self,b,a,l,n):
        x = G.SO3partArr.randn(b,[a],l,n)
        y = G.SO3partArr.randn(b,[a],l,n)
        z=G.CGproduct(x,y,l)

        R = G.SO3element.uniform()
        xr=x.rotate(R)
        yr=y.rotate(R)
        zr=G.CGproduct(xr,yr,l)
        rz=z.rotate(R)

        torch.allclose(rz,zr,rtol=1e-3, atol=1e-4)

    @pytest.mark.parametrize('b', [1, 2, 4])    
    @pytest.mark.parametrize('a', [1, 2, 4])    
    @pytest.mark.parametrize('l', [1, 2, 4, 8])
    @pytest.mark.parametrize('n', [1, 2, 4, 8, 32])
    def test_DiagCGproduct(self,b,a,l,n):
        x=G.SO3partArr.randn(b,[a],l,n)
        y=G.SO3partArr.randn(b,[a],l,n)
        z=G.DiagCGproduct(x,y,l)

        R=G.SO3element.uniform()
        xr=x.rotate(R)
        yr=y.rotate(R)
        zr=G.DiagCGproduct(xr,yr,l)
        rz=z.rotate(R)

        assert torch.allclose(rz,zr,rtol=1e-3, atol=1e-5)


    @pytest.mark.parametrize('b', [1, 2, 4])    
    @pytest.mark.parametrize('a', [1, 2, 4])    
    @pytest.mark.parametrize('l', [1, 2, 4, 8])
    @pytest.mark.parametrize('n', [1, 2, 4, 8, 32])
    def test_CGproduct_backprop(self,b,a,l,n):
        self.partArr_partArr_backprop(b,[a],l,n,G.CGproduct,l)

