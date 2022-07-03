import torch
import gelib as G
import pytest

class TestSO3mvec(object):
    
    @pytest.mark.parametrize('tau', [1, 2, 4, 8, 32])
    @pytest.mark.parametrize('maxl', range(6))
    @pytest.mark.parametrize('b', [1, 2, 3])    
    @pytest.mark.parametrize('k', [1, 2, 4])    
    def test_CGproduct(self,b,k,tau,maxl):
        x = G.SO3mvec.randn(b,k,[tau for i in range(maxl + 1)])
        y = G.SO3mvec.randn(b,k,[tau for i in range(maxl + 1)])
        R = G.SO3element.uniform()
        xr=x.rotate(R)
        yr=y.rotate(R)
        z=G.CGproduct(x,y,maxl=maxl)
        #print(z)
        zr=G.CGproduct(xr,yr,maxl=maxl)
        rz=z.rotate(R)

        for i in range(maxl+1 ):
            assert (torch.allclose(rz.parts[i] , zr.parts[i], rtol=1e-3, atol=1e-4))


    @pytest.mark.parametrize('tau', [1, 2, 4, 8, 32])
    @pytest.mark.parametrize('maxl', range(7))
    @pytest.mark.parametrize('b', [1, 2, 4])    
    @pytest.mark.parametrize('k', [1, 2, 4])    
    def test_DiagCGproduct(self,b,k,tau, maxl):
        x=G.SO3mvec.randn(b,k,[tau for i in range(maxl + 1)])
        y=G.SO3mvec.randn(b,k,[tau for i in range(maxl + 1)])
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
    @pytest.mark.parametrize('k', [1, 2, 4])    
    def test_Fproduct(self,b,k,maxl):
        x=G.SO3mvec.Frandn(b,k,maxl)
        y=G.SO3mvec.Frandn(b,k,maxl)
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
    @pytest.mark.parametrize('k', [1, 2, 4])    
    def test_CGproduct_backprop(self,b,k,tau, maxl):
        return
        x = G.SO3mvec.randn(b,k,[tau for i in range(maxl + 1)])
        y = G.SO3mvec.randn(b,k,[tau for i in range(maxl + 1)])
        z=G.CGproduct(x,y,maxl=maxl)

        zr=G.CGproduct(xr,yr,maxl=maxl)


