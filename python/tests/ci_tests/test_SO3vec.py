import torch
import gelib as G
import pytest

class TestSO3vec(object):
    
    def vec_vec_backprop(self,b,tau,fn):
        x = G.SO3vec.randn(b,tau)
        y = G.SO3vec.randn(b,tau)
        x.requires_grad_()
        y.requires_grad_()
        z=fn(x,y)

        test_vec=G.SO3vec.randn_like(z)
        loss=z.odot(test_vec)
        loss.backward(torch.tensor(1.0))
        xgrad=x.get_grad()
        ygrad=y.get_grad()

        xeps=G.SO3vec.randn_like(x)
        z=fn(x+xeps,y)
        xloss=z.odot(test_vec)
        assert(torch.allclose(xloss-loss,xeps.odot(xgrad),rtol=1e-3, atol=1e-4))

        yeps=G.SO3vec.randn_like(x)
        z=fn(x,y+yeps)
        yloss=z.odot(test_vec)
        assert(torch.allclose(yloss-loss,yeps.odot(ygrad),rtol=1e-3, atol=1e-4))
        

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
        self.vec_vec_backprop(b,[tau for i in range(maxl + 1)],G.CGproduct)
        return

    @pytest.mark.parametrize('tau', [1, 2, 4, 8, 32])
    @pytest.mark.parametrize('maxl', range(7))
    @pytest.mark.parametrize('b', [1, 2, 4])    
    def test_DiagCGproduct_backprop(self,b,tau, maxl):
        self.vec_vec_backprop(b,[tau for i in range(maxl + 1)],G.DiagCGproduct)
        return

    @pytest.mark.parametrize('maxl', range(7))
    @pytest.mark.parametrize('b', [1, 2, 4])    
    def test_Fproduct_backprop(self,b,maxl):
        self.vec_vec_backprop(b,[2*l+1 for l in range(maxl + 1)],G.Fproduct)
        return
