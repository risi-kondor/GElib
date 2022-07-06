import torch
import gelib as G
import pytest

class TestSO3vec(object):
    
    @pytest.mark.parametrize('tau', [1, 2, 4, 8, 32])
    @pytest.mark.parametrize('maxl', range(7))
    @pytest.mark.parametrize('b', [1, 2, 4])    
    def test_CGproduct(self,b,tau, maxl):
        x = G.SO3vec.randn(b,[tau for i in range(maxl + 1)])
        y = G.SO3vec.randn(b,[tau for i in range(maxl + 1)])
        R = G.SO3element.uniform()
        xr=x.rotate(R)
        yr=y.rotate(R)

        z=G.CGproduct(x,y,maxl=maxl)
        zr=G.CGproduct(xr,yr,maxl=maxl)
        rz=z.rotate(R)

        for i in range(maxl+1 ):
            assert (torch.allclose(rz.parts[i] , zr.parts[i], rtol=1e-3, atol=1e-4))


    @pytest.mark.parametrize('tau', [1, 2, 4, 8, 32])
    @pytest.mark.parametrize('maxl', range(7))
    @pytest.mark.parametrize('b', [1, 2, 4])    
    def test_DiagCGproduct(self,b,tau, maxl):
        x=G.SO3vec.randn(b,[tau for i in range(maxl + 1)])
        y=G.SO3vec.randn(b,[tau for i in range(maxl + 1)])
        R=G.SO3element.uniform()
        xr=x.rotate(R)
        yr=y.rotate(R)

        z=G.DiagCGproduct(x,y,maxl=maxl)
        zr=G.DiagCGproduct(xr,yr,maxl=maxl)
        rz=z.rotate(R)

        for i in range(maxl+1 ):
            assert (torch.allclose(rz.parts[i] , zr.parts[i], rtol=1e-3, atol=1e-5))

    @pytest.mark.parametrize('maxl', range(7))
    @pytest.mark.parametrize('b', [1, 2, 4])    
    def test_Fproduct(self,b,maxl):
        x=G.SO3vec.Frandn(b,maxl)
        y=G.SO3vec.Frandn(b,maxl)
        R = G.SO3element.uniform()
        xr=x.rotate(R)
        yr=y.rotate(R)

        z=G.Fproduct(x,y,maxl=maxl)
        zr=G.Fproduct(xr,yr,maxl=maxl)
        rz=z.rotate(R)

        for i in range(maxl+1 ):
            assert (torch.allclose(rz.parts[i] , zr.parts[i], rtol=1e-3, atol=1e-4))


    @pytest.mark.parametrize('tau', [1, 2, 4, 8, 32])
    @pytest.mark.parametrize('maxl', range(7))
    @pytest.mark.parametrize('b', [1, 2, 4])    
    def test_CGproduct_backprop(self,b,tau, maxl):
        return
        x = G.SO3vec.randn(b,[tau for i in range(maxl + 1)])
        y = G.SO3vec.randn(b,[tau for i in range(maxl + 1)])
        x.requires_grad_()
        y.requires_grad_()
        torch.autograd.gradcheck(G.SO3vec_CGproductFn.apply,[maxl+1,maxl+1,maxl,*(x.parts+y.parts)])

