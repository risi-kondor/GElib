import torch
import gelib as G
import pytest

class TestSO3vecArr(object):
    
    def vec_vec_backprop(self,b,A,tau,fn):
        x = G.SO3vecArr.randn(b,A,tau)
        y = G.SO3vecArr.randn(b,A,tau)
        x.requires_grad_()
        y.requires_grad_()
        z=fn(x,y)

        test_vec=G.SO3vecArr.randn_like(z)
        loss=z.odot(test_vec)
        loss.backward(torch.tensor(1.0))
        xgrad=x.get_grad()
        ygrad=y.get_grad()

        xeps=G.SO3vecArr.randn_like(x)
        z=fn(x+xeps,y)
        xloss=z.odot(test_vec)
        assert(torch.allclose(xloss-loss,xeps.odot(xgrad),rtol=1e-3, atol=1e-4))
        return

        yeps=G.SO3vecArr.randn_like(x)
        z=fn(x,y+yeps)
        yloss=z.odot(test_vec)
        assert(torch.allclose(yloss-loss,yeps.odot(ygrad),rtol=1e-3, atol=1e-4))
        

    def vec_vec_backprop_bcast(self,b,A,tau,fn):
        x = G.SO3vecArr.randn(1,A,tau)
        y = G.SO3vecArr.randn(b,A,tau)
        x.requires_grad_()
        y.requires_grad_()
        z=fn(x,y)

        test_vec=G.SO3vecArr.randn_like(z)
        loss=z.odot(test_vec)
        loss.backward(torch.tensor(1.0))
        xgrad=x.get_grad()
        ygrad=y.get_grad()

        xeps=G.SO3vecArr.randn_like(x)
        z=fn(x+xeps,y)
        xloss=z.odot(test_vec)
        assert(torch.allclose(xloss-loss,xeps.odot(xgrad),rtol=1e-3, atol=1e-4))

        yeps=G.SO3vecArr.randn_like(x)
        z=fn(x,y+yeps)
        yloss=z.odot(test_vec)
        assert(torch.allclose(yloss-loss,yeps.odot(ygrad),rtol=1e-3, atol=1e-4))
        

    @pytest.mark.parametrize('b', [1, 2, 4])    
    @pytest.mark.parametrize('a', [2])    
    @pytest.mark.parametrize('nc', [1, 2, 4])
    @pytest.mark.parametrize('maxl', [2,4])
    def test_CGproduct(self,b,a,nc,maxl):
        tau={l:nc for l in range(maxl+1)}
        x = G.SO3vecArr.randn(b,[a,a],tau)
        y = G.SO3vecArr.randn(b,[a,a],tau)
        R = G.SO3element.random()
        xr=x.apply(R)
        yr=y.apply(R)

        z=G.CGproduct(x,y,maxl=maxl)
        zr=G.CGproduct(xr,yr,maxl=maxl)
        rz=z.apply(R)

        for i in range(maxl+1):
            assert (torch.allclose(rz.parts[i] , zr.parts[i], rtol=1e-3, atol=1e-4))


    @pytest.mark.parametrize('b', [1, 3])    
    @pytest.mark.parametrize('a', [2])    
    @pytest.mark.parametrize('nc', [1, 2, 4])
    @pytest.mark.parametrize('maxl',[2,4])
    def test_DiagCGproduct(self,b,a,nc,maxl):
        tau={l:nc for l in range(maxl+1)}
        x=G.SO3vecArr.randn(b,[a,a],tau)
        y=G.SO3vecArr.randn(b,[a,a],tau)
        R=G.SO3element.random()
        xr=x.apply(R)
        yr=y.apply(R)

        z=G.DiagCGproduct(x,y,maxl=maxl)
        zr=G.DiagCGproduct(xr,yr,maxl=maxl)
        rz=z.apply(R)

        for i in range(maxl+1):
            assert (torch.allclose(rz.parts[i] , zr.parts[i], rtol=1e-3, atol=1e-4))


    @pytest.mark.parametrize('b', [1,3])    
    @pytest.mark.parametrize('a', [2])    
    @pytest.mark.parametrize('nc', [1,2,4])
    @pytest.mark.parametrize('maxl',[2,4])
    def test_CGproduct_backprop(self,b,a,nc,maxl):
        tau={l:nc for l in range(maxl+1)}
        self.vec_vec_backprop(b,[a,a],tau,G.CGproduct)
        return

    @pytest.mark.parametrize('b', [1, 3])    
    @pytest.mark.parametrize('a', [2])    
    @pytest.mark.parametrize('nc', [1, 4])
    @pytest.mark.parametrize('maxl',[2,4])
    def test_DiagCGproduct_backprop(self,b,a,nc,maxl):
        tau={l:nc for l in range(maxl+1)}
        self.vec_vec_backprop(b,[a,a],tau,G.DiagCGproduct)
        return

    @pytest.mark.parametrize('b', [1, 3])    
    @pytest.mark.parametrize('a', [2])    
    @pytest.mark.parametrize('nc', [1, 4])
    @pytest.mark.parametrize('maxl',[2,4])
    def test_CGproduct_backprop_bcast(self,b,a,nc,maxl):
        tau={l:nc for l in range(maxl+1)}
        self.vec_vec_backprop_bcast(b,[a,a],tau,G.CGproduct)
        return

