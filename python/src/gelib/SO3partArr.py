
# This file is part of GElib, a C++/CUDA library for group
# equivariant tensor operations. 
# 
# Copyright (c) 2022, Imre Risi Kondor
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


from typing import List
import torch
from cnine import rtensor as _rtensor
from cnine import ctensor as _ctensor
from gelib_base import SO3partB_array as _SO3partB_array
from gelib_base import add_conterpolate3dB, add_conterpolate3dB_back

class SO3partArr(torch.Tensor):
    """
    A collection of vectors that transform according to a specific irreducible representation of SO(3).
    The vectors are stacked into a fourth order tensor. The first index is the site index, the second index
    is the batch index, the third is m=-l,...,l, and the fourth index is the fragment index. 
    """
    def __new__(cls, x : torch.Tensor, *args, **kwargs):
        if(torch.is_complex(x)):
            return torch.view_as_complex(super().__new__(cls, torch.view_as_real(x), *args, **kwargs))
        else:
            return torch.view_as_complex(super().__new__(cls,x, *args, **kwargs))
        
    #@classmethod 
    #def __init__(self,_T):
        #if(torch.is_complex(_T)):
            #self=torch.view_as_real(_T)
            #super().__init__(torch.view_as_real(_T))
        #else:
            #super().__init__(_T)
            #self=_T


    ## ---- Static constructors -----------------------------------------------------------------------------

    
    @staticmethod
    def zeros(b : int, _adims : List[int], l : int, n : int, device : str = 'cpu') -> 'SO3partArr':
        """
        Create an SO(3)-part consisting of N*b lots of n vectors transforming according to the l'th irrep of SO(3).
        The vectors are initialized to zero, resulting in an b*(2+l+1)*n dimensional complex tensor of zeros.
        """
        if _adims == None:
            _adims = []
        return SO3partArr(torch.zeros([b]+_adims+[2*l+1,n,2],device=device))
    
    @staticmethod
    def ones(b : int, _adims : List[int], l : int, n : int, device : str = 'cpu') -> 'SO3partArr':
        """
        Create an SO(3)-part consisting of N*b lots of n vectors transforming according to the l'th irrep of SO(3).
        The vectors are initialized to one, resulting in an b*(2+l+1)*n dimensional complex tensor of ones.
        """        
        if _adims == None:
            _adims = []
        ones = torch.ones([b]+_adims+[2*l+1,n],device=device)
        zeros = torch.zeros([b]+_adims+[2*l+1,n],device=device)
        return SO3partArr(torch.stack([ones, zeros], dim = -1))

    @staticmethod
    def randn(b : int, _adims : List[int], l : int, n : int, device : str = 'cpu') -> 'SO3partArr':
        """
        Create an SO(3)-part consisting of N*b lots of n vectors transforming according to the l'th irrep of SO(3).
        The vectors are initialized as random gaussian vectors, resulting in an b*(2+l+1)*n dimensional random
        complex tensor.
        """
        if _adims == None:
            _adims = []
        return SO3partArr(torch.randn([b]+_adims+[2*l+1,n,2],device=device))
        #return torch.view_as_complex(SO3partArr(torch.randn([b]+_adims+[2*l+1,n,2],device=device)))

    @classmethod
    def spharm(self, l : int, X : torch.Tensor, device : str = 'cpu') -> 'SO3partArr':
        """
        Return the spherical harmonics of the vector (x,y,z)
        """
        R =SO3partArr.zeros(X.size(0),list(X.size())[1:X.dim()-2], l, X.size(-1), device=device)
        _SO3partB_array.view(R).add_spharm(X)
        return R.to(device)

    @staticmethod
    def Fzeros(b : int, _adims : List[int], l : int, device : str = 'cpu') -> 'SO3partArr':
        """
        Create an SO(3)-part corresponding to the l'th matrix in the Fourier transform of a function on SO(3).
        This gives a N*b*(2+l+1)*(2l+1) dimensional complex tensor. 
        """
        if _adims == None:
            _adims = []
            
        return torch.view_as_complex(SO3partArr(torch.zeros([b]+_adims+[2*l+1,2*l+1,2],device=device)))


    @staticmethod
    def Frandn(b : int, _adims : List[int], l : int, device : str = 'cpu') -> 'SO3partArr':
        """
        Create an SO(3)-part corresponding to the l'th matrix in the Fourier transform of a function on SO(3).
        This gives a b*(2+l+1)*(2l+1) dimensional complex random tensor. 
        """
        if _adims == None:
            _adims = []
            
        return torch.view_as_complex(SO3partArr(torch.randn([b]+_adims+[2*l+1,2*l+1,2],device=device)))


    @classmethod
    def zeros_like(self, x : 'SO3partArr'):
        return torch.view_as_complex(SO3partArr(torch.zeros_like(torch.view_as_real(x))))
    
    @classmethod
    def randn_like(self, x : 'SO3partArr'):
        return SO3partArr(torch.randn_like(torch.view_as_real(x)))


    ## ---- Access ------------------------------------------------------------------------------------------


    def getb(self) -> int:
        return self.size(0)

    def get_adims(self) -> List[int]:
        return list(self.size()[1:self.dim()-2])

    def get_nadims(self) -> int:
        return self.dim()-3

    def getl(self) -> int:
        return int((self.size(-2)-1)/2)

    def getn(self) -> int:
        return self.size(-1)


    ## ---- Operations --------------------------------------------------------------------------------------


    def odot(self,y) -> int:
            return torch.sum(torch.mul(torch.view_as_real(self),torch.view_as_real(y)))

    def rotate(self,R) -> 'SO3partArr':
        A= _SO3partB_array.view(self).rotate(R).torch()
        return SO3partArr(torch.view_as_real(A))
        #return torch.view_as_complex(SO3partArr(torch.view_as_real(A)))

    def gather(self,_mask) -> 'SO3partArr':
        """
        Gather the elements of this SO3partArr into a new SO3partArr according to the mask
        """
        return SO3partArr_GatherFn.apply(_mask,self)

    def conterpolate(self,M) -> 'SO3partArr':
        return SO3partArr_ConterpolateFn.apply(self,M)

    def conterpolateB(self,M) -> 'SO3partArr':
        return SO3partArr_ConterpolateBFn.apply(self,M)


    # ---- Products -----------------------------------------------------------------------------------------


    def CGproduct(self, y : 'SO3partArr', l : int) -> 'SO3partArr':
        """
        Compute the l component of the Clesbsch--Gordan product of this SO3partArr with another SO3partArr y.
        """
        assert y != None
        assert isinstance(y, SO3partArr)
        assert isinstance(l, int)
        return SO3partArr_CGproductFn.apply(self,y,l)


    def DiagCGproduct(self, y : 'SO3partArr', l : int) -> 'SO3partArr':
        """
        Compute the l component of the diagonal Clesbsch--Gordan product of this SO3partArr with another SO3partArr y.
        """
        assert y != None
        assert isinstance(y, SO3partArr)
        assert isinstance(l, int)
        return SO3partArr_DiagCGproductFn.apply(self,y,l)


    ## ---- I/O ----------------------------------------------------------------------------------------------

        
    def __repr__(self):
        u=_SO3partB_array.view(self)
        return u.__repr__()

    def __str__(self):
        u=_SO3partB_array.view(self)
        return u.__str__()



## ----------------------------------------------------------------------------------------------------------
## ---- Autograd functions -----------------------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------------------


class SO3partArr_CGproductFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y,l):
        ctx.l=l
        ctx.save_for_backward(x,y)

        r = SO3partArr.zeros(x.getb(),x.get_adims(),l,x.getn()*y.getn(),x.device)

        _x = _SO3partB_array.view(x)
        _y = _SO3partB_array.view(y)
        _r = _SO3partB_array.view(r)
        _r.addCGproduct(_x,_y)

        return r

    @staticmethod
    def backward(ctx, g):

        x,y = ctx.saved_tensors

        xg=torch.zeros_like(x)
        yg=torch.zeros_like(y)

        _x = _SO3partB_array.view(x)
        _y = _SO3partB_array.view(y)

        _g = _SO3partB_array.view(g)
        _xg = _SO3partB_array.view(xg)
        _yg = _SO3partB_array.view(yg)

        _xg.addCGproduct_back0(_g, _y)
        _yg.addCGproduct_back1(_g, _x)

        return xg,yg,None


class SO3partArr_DiagCGproductFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y,l):
        ctx.l=l
        assert x.size(2)==y.size(2)
        ctx.save_for_backward(x,y)

        r = SO3partArr.zeros(x.getb(),x.get_adims(),l,x.getn(),x.device)

        _x = _SO3partB_array.view(x)
        _y = _SO3partB_array.view(y)
        _r = _SO3partB_array.view(r)
        _r.addDiagCGproduct(_x,_y)

        return r

    @staticmethod
    def backward(ctx, g):

        x,y = ctx.saved_tensors

        xg=torch.zeros_like(x)
        yg=torch.zeros_like(y)

        _x = _SO3partB_array.view(x)
        _y = _SO3partB_array.view(y)

        _g = _SO3partB_array.view(g)
        _xg = _SO3partB_array.view(xg)
        _yg = _SO3partB_array.view(yg)

        _xg.addDiagCGproduct_back0(_g, _y)
        _yg.addDiagCGproduct_back1(_g, _x)

        return xg,yg,None


class SO3partArr_GatherFn(torch.autograd.Function): 

    @staticmethod
    def forward(ctx,_mask,x):

        ctx.mask=_mask
        l=x.getl()
        n=x.getn()
        r=SO3partArr.zeros(x.getb(),x.get_adims(),l,n,x.device) # TODO
        
        _x=_SO3partB_array.view(x)
        _r=_SO3partB_array.view(r)
        _r.add_gather(_x,_mask)

        return r

    @staticmethod
    def backward(ctx,yg,*args):

        #N=yg.size(0)
        #b=yg.size(1)
        #l=int((y.size(2)-1)/2)
        #n=y.size(3)
        #dev=int(y.is_cuda)
        #r=MakeZeroSO3partArrs(N,b,l,n,dev)
        r=torch.zeros_like(yg) # change this

        # TODO: What is args?
        _x=_SO3partB_array.view(*args)
        _r=_SO3partB_array.view(r)
        _r.gather(_x,ctx.mask.inv())

        return tuple([None]+r)


class SO3partArr_ConterpolateFn(torch.autograd.Function): 

    @staticmethod
    def forward(ctx,x,M):
        ctx.nadims=x.get_nadims()
        ctx.M=M
        _x=_SO3partB_array.view(x)
        _M=_rtensor.view(M)
        if(x.get_nadims()==2):
            r=SO3partArr.zeros(x.getb(),x.get_adims()+list(M.size()[:-2]),x.getl(),x.getn(),x.device)
            _r=_SO3partB_array.view(r)
            _r.add_conterpolate2d(_x,_M)
        if(x.get_nadims()==3):
            r=SO3partArr.zeros(x.getb(),x.get_adims()+list(M.size()[:-3]),x.getl(),x.getn(),x.device)
            print(r.size())
            _r=_SO3partB_array.view(r)
            _r.add_conterpolate3d(_x,_M)
        return r

    @staticmethod
    def backward(ctx,g):
        _g=_SO3partB_array.view(g)
        _M=_rtensor.view(ctx.M)
        gx=SO3partArr.zeros(g.getb(),g.get_adims()[:ctx.nadims],g.getl(),g.getn(),g.device)
        _gx=_SO3partB_array.view(gx)
        if(ctx.nadims==2):
            _gx.add_conterpolate2d_back(_g,_M)
        if(ctx.nadims==3):
            _gx.add_conterpolate3d_back(_g,_M)
        return gx,None


class SO3partArr_ConterpolateBFn(torch.autograd.Function): 

    @staticmethod
    def forward(ctx,x,M):
        ctx.b=x.getb()
        ctx.adims=x.get_adims()
        ctx.l=x.getl()
        ctx.n=x.getn()
        ctx.M=M
        _x=_SO3partB_array.view(x)
        _M=_rtensor.view(M)
        r=torch.zeros([x.getb()]+x.get_adims()+list(M.size()[:-4])+[x.getn()],dtype=torch.cfloat,device=x.device)
        _r=_ctensor.view(r)
        add_conterpolate3dB(_r,_x,_M)
        return r

    @staticmethod
    def backward(ctx,g):
        _g=_ctensor.view(g)
        _M=_rtensor.view(ctx.M)
        gx=SO3partArr.zeros(ctx.b,ctx.adims,ctx.l,ctx.n,g.device)
        print(gx.size())
        _gx=_SO3partB_array.view(gx)
        add_conterpolate3dB_back(_gx,_g,_M)
        return gx,None



## ----------------------------------------------------------------------------------------------------------
## ---- Helpers ---------------------------------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------------------


def CGproduct(x,y,maxl=-1):
    return x.CGproduct(y,maxl)
    


