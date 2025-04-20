import torch
import gelib as G
import pytest

class TestSO3partArr(object):
    
    def part_part_backprop(self,b,A,l,n,fn,arg0):
        x = G.SO3partArr.randn(b,A,l,n)
        y = G.SO3partArr.randn(b,A,l,n)
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


    def part_part_backprop_bcast(self,b,A,l,n,fn,arg0):
        x = G.SO3partArr.randn(1,A,l,n)
        y = G.SO3partArr.randn(b,A,l,n)
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


    @pytest.mark.parametrize('b', [1, 2])    
    @pytest.mark.parametrize('a', [2])    
    @pytest.mark.parametrize('l', [1, 2, 4])
    @pytest.mark.parametrize('n', [1, 2, 4, 8])
    def test_CGproduct(self,b,a,l,n):
        x = G.SO3partArr.randn(b,[a,a],l,n)
        y = G.SO3partArr.randn(b,[a,a],l,n)
        z=G.CGproduct(x,y,l)

        R = G.SO3element.random()
        xr=x.apply(R)
        yr=y.apply(R)
        zr=G.CGproduct(xr,yr,l)
        rz=z.apply(R)

        assert torch.allclose(rz,zr,rtol=1e-3, atol=1e-4)


    @pytest.mark.parametrize('b', [1, 2])    
    @pytest.mark.parametrize('a', [2])    
    @pytest.mark.parametrize('l', [1, 2, 4])
    @pytest.mark.parametrize('n', [1, 2, 4, 8])
    def test_DiagCGproduct(self,b,a,l,n):
        x=G.SO3partArr.randn(b,[a,a],l,n)
        y=G.SO3partArr.randn(b,[a,a],l,n)
        z=G.DiagCGproduct(x,y,l)

        R=G.SO3element.random()
        xr=x.apply(R)
        yr=y.apply(R)
        zr=G.DiagCGproduct(xr,yr,l)
        rz=z.apply(R)

        assert torch.allclose(rz,zr,rtol=1e-3, atol=1e-5)


    @pytest.mark.parametrize('b', [1, 2])
    @pytest.mark.parametrize('a', [2])    
    @pytest.mark.parametrize('l', [1, 2, 4])
    @pytest.mark.parametrize('n', [1, 2, 4, 8])
    def test_CGproduct_backprop(self,b,a,l,n):
        self.part_part_backprop(b,[a,a],l,n,G.CGproduct,l)


    @pytest.mark.parametrize('b', [1, 2])    
    @pytest.mark.parametrize('a', [2])    
    @pytest.mark.parametrize('l', [1, 2, 4])
    @pytest.mark.parametrize('n', [1, 2, 4, 8])
    def test_DiagCGproduct_backprop(self,b,a,l,n):
        self.part_part_backprop(b,[a,a],l,n,G.DiagCGproduct,l)


    @pytest.mark.parametrize('b', [1, 2])    
    @pytest.mark.parametrize('a', [2])    
    @pytest.mark.parametrize('l', [2, 4])
    @pytest.mark.parametrize('n', [4])
    def test_CGproduct_backprop_bcast(self,b,a,l,n):
        self.part_part_backprop_bcast(b,[a,a],l,n,G.CGproduct,l)




