import torch
import gelib as G
import pytest

class TestSO3part(object):
    
    def part_part_backprop(self,b,l,n,fn,arg0):
        x = G.SO3part.randn(b,l,n)
        y = G.SO3part.randn(b,l,n)
        x.requires_grad_()
        y.requires_grad_()
        z=fn(x,y,arg0)

        test_vec=G.SO3part.randn_like(z)
        loss=z.odot(test_vec)
        loss.backward(torch.tensor(1.0))
        xgrad=x.grad
        ygrad=y.grad

        xeps=G.SO3part.randn_like(x)
        z=fn(x+xeps,y,arg0)
        xloss=z.odot(test_vec)
        assert(torch.allclose(xloss-loss,xeps.odot(xgrad),rtol=1e-3, atol=1e-4))

        yeps=G.SO3part.randn_like(x)
        z=fn(x,y+yeps,arg0)
        yloss=z.odot(test_vec)
        assert(torch.allclose(yloss-loss,yeps.odot(ygrad),rtol=1e-3, atol=1e-4))


    @pytest.mark.parametrize('b', [1, 2])    
    @pytest.mark.parametrize('l', [1, 2, 4])
    @pytest.mark.parametrize('n', [1, 2, 4, 8])
    def test_CGproduct(self,b,l,n):
        x = G.SO3part.randn(b,l,n)
        y = G.SO3part.randn(b,l,n)
        z=G.CGproduct(x,y,l)

#         R = G.SO3element.uniform()
#         xr=x.rotate(R)
#         yr=y.rotate(R)
#         zr=G.CGproduct(xr,yr,l)
#         rz=z.rotate(R)

#         torch.allclose(rz,zr,rtol=1e-3, atol=1e-4)


    @pytest.mark.parametrize('b', [1, 2])    
    @pytest.mark.parametrize('l', [1, 2, 4])
    @pytest.mark.parametrize('n', [1, 2, 4, 8])
    def test_DiagCGproduct(self,b,l,n):
        x=G.SO3part.randn(b,l,n)
        y=G.SO3part.randn(b,l,n)
        z=G.DiagCGproduct(x,y,l)

#         R=G.SO3element.uniform()
#         xr=x.rotate(R)
#         yr=y.rotate(R)
#         zr=G.DiagCGproduct(xr,yr,l)
#         rz=z.rotate(R)

#         assert torch.allclose(rz,zr,rtol=1e-3, atol=1e-5)


    @pytest.mark.parametrize('b', [1, 2])    
    @pytest.mark.parametrize('l', [1, 2, 4])
    @pytest.mark.parametrize('n', [1, 2, 4, 8])
    def test_CGproduct_backprop(self,b,l,n):
        self.part_part_backprop(b,l,n,G.CGproduct,l)


    @pytest.mark.parametrize('b', [1, 2])    
    @pytest.mark.parametrize('l', [1, 2, 4])
    @pytest.mark.parametrize('n', [1, 2, 4, 8])
    def test_DiagCGproduct_backprop(self,b,l,n):
        self.part_part_backprop(b,l,n,G.DiagCGproduct,l)


