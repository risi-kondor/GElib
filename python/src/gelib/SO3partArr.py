
# This file is part of GElib, a C++/CUDA library for group
# equivariant tensor operations. 
# 
# Copyright (c) 2022, Imre Risi Kondor
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


import torch
import gelib_base as gb
import gelib
from gelib import *


class SO3partArr(torch.Tensor):
    """
    A collection of vectors that transform according to a specific irreducible representation of SO(3).
    The vectors are stacked into a fourth order tensor. The first index is the site index, the second index
    is the batch index, the third is m=-l,...,l, and the fourth index is the fragment index. 
    """
    def __init__(self, _T):
        self=_T


    ## ---- Static constructors -----------------------------------------------------------------------------

    
    @staticmethod
    def zeros(b,adims,l,n,device='cpu'):
        """
        Create an SO(3)-part consisting of N*b lots of n vectors transforming according to the l'th irrep of SO(3).
        The vectors are initialized to zero, resulting in an b*(2+l+1)*n dimensional complex tensor of zeros.
        """        
        return SO3partArr(torch.zeros([b]+adims+[2*l+1,n],dtype=torch.complex64,device=device))

    
    @staticmethod
    def ones(b,adims,l,n,device='cpu'):
        """
        Create an SO(3)-part consisting of N*b lots of n vectors transforming according to the l'th irrep of SO(3).
        The vectors are initialized to one, resulting in an b*(2+l+1)*n dimensional complex tensor of ones.
        """        
        return SO3partArr(torch.ones([b]+adims+[2*l+1,n],dtype=torch.complex64,device=device))


    @staticmethod
    def randn(b,adims,l,n,device='cpu'):
        """
        Create an SO(3)-part consisting of N*b lots of n vectors transforming according to the l'th irrep of SO(3).
        The vectors are initialized as random gaussian vectors, resulting in an b*(2+l+1)*n dimensional random
        complex tensor.
        """
        return SO3partArr(torch.randn([b]+adims+[2*l+1,n],dtype=torch.complex64,device=device))


    @classmethod
    def spharm(self, l, X, device='cpu'):
        """
        Return the spherical harmonics of the vector (x,y,z)
        """
        assert(X.size(-2)==3)
        R=SO3partArr.zeros(X.size(0),list(X.size())[1:X.dim()-2],l,X.size(-1),device='cpu')
        R.backend().add_spharm(X)
        return R.to(device)


    @staticmethod
    def Fzeros(b,adims,l,device='cpu'):
        """
        Create an SO(3)-part corresponding to the l'th matrix in the Fourier transform of a function on SO(3).
        This gives a N*b*(2+l+1)*(2l+1) dimensional complex tensor. 
        """
        return SO3partArr(torch.zeros([b]+adims+[2*l+1,2*l+1],dtype=torch.complex64,device=device))


    @staticmethod
    def Frandn(b,adims,l,device='cpu'):
        """
        Create an SO(3)-part corresponding to the l'th matrix in the Fourier transform of a function on SO(3).
        This gives a b*(2+l+1)*(2l+1) dimensional complex random tensor. 
        """
        return SO3partArr(torch.randn([b]+adims+[2*l+1,2*l+1],dtype=torch.complex64,device=device))


    def zeros_like(self,*args):
        if not args:
            return SO3partArr(torch.zeros_like(self))
        if len(args)==2:
            assert isinstance(args[0],int)
            assert isinstance(args[1],int)
            dims=list(self.size())
            dims[-2]=2*args[0]+1
            dims[-1]=args[1]
            return SO3partArr(torch.zeros(dims,dtype=torch.complex64,device=self.device))
    
    
    @classmethod
    def randn_like(self,x):
        return SO3partArr(torch.randn_like(x))


    def backend(self):
        return gb.SO3part.view(self)

    def collapse(self):
        return SO3part(self.reshape(self.size(0),-1,self.size(-2),self.size(-1)))


    ## ---- Access ------------------------------------------------------------------------------------------


    def getb(self):
        return self.size(0)

    def get_adims(self):
        return list(self.size()[1:self.dim()-2])

    def get_nadims(self):
        return self.dim()-3

    def getl(self):
        return int((self.size(-2)-1)/2)

    def getn(self):
        return self.size(-1)


    ## ---- Operations --------------------------------------------------------------------------------------


    def odot(self,y):
        return torch.Tensor(torch.sum(torch.mul(torch.view_as_real(self),torch.view_as_real(y))))

    def apply(self, R):
        assert(isinstance(R,SO3element))
        rho=SO3irrep(self.getl())
        return SO3partArr(torch.matmul(rho.matrix(R),self.collapse()).reshape(self.size()))

    def gather(self,gmap,dim=0):
        """
        Gather the cells of this SO3partArr into a new SO3partArr according to the gather_map
        """
        return SO3partArr_GatherFn.apply(self,gmap,dim)

#     def conterpolate(self,M):
#         return SO3partArr_ConterpolateFn.apply(self,M)

#     def conterpolateB(self,M):
#         return SO3partArr_ConterpolateBFn.apply(self,M)


    # ---- Products -----------------------------------------------------------------------------------------


    def CGproduct(self,y,l):
        """
        Compute the l component of the Clesbsch--Gordan product of this SO3partArr with another SO3partArr y.
        """
        return gelib.SO3partArr_CGproductFn.apply(self,y,l)


    def DiagCGproduct(self,y,l):
        """
        Compute the l component of the diagonal Clesbsch--Gordan product of this SO3partArr with another SO3partArr y.
        """
        return gelib.SO3partArr_DiagCGproductFn.apply(self,y,l)


    ## ---- I/O ----------------------------------------------------------------------------------------------

        
    def __repr__(self):
        return self.backend().__repr__()

    def __str__(self):
        return self.backend().__str__()


## ----------------------------------------------------------------------------------------------------------
## ---- Autograd functions -----------------------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------------------


class SO3partArr_CGproductFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y,l):
        ctx.l=l
        ctx.save_for_backward(x,y)
        #b=max(x.size(0),y.size(0))
        #r=x.zeros_like(l,x.size(-1)*y.size(-1))
        b=common_batch(x,y)
        a=x.get_adims()
        r=SO3partArr.zeros(b,a,l,x.size(-1)*y.size(-1),device=x.device)
        r.backend().add_CGproduct(x.backend(),y.backend())
        return r

    @staticmethod
    def backward(ctx,g):
        x,y = ctx.saved_tensors
        #g=SO3part(_g)
        xg=x.zeros_like()
        yg=y.zeros_like()
        gb.SO3part.view(xg).add_CGproduct_back0(gb.SO3part.view(g),gb.SO3part.view(y))
        gb.SO3part.view(yg).add_CGproduct_back1(gb.SO3part.view(g),gb.SO3part.view(x))
        return xg,yg,None


class SO3partArr_DiagCGproductFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y,l):
        ctx.l=l
        ctx.save_for_backward(x,y)
        #b=max(x.size(0),y.size(0))
        #r=x.zeros_like(l,x.size(-1))
        b=common_batch(x,y)
        a=x.get_adims()
        assert x.size(-1)==y.size(-1)
        r=SO3partArr.zeros(b,a,l,x.size(-1),device=x.device)
        r.backend().add_DiagCGproduct(x.backend(),y.backend())
        return r

    @staticmethod
    def backward(ctx,g):
        x,y = ctx.saved_tensors
        xg=x.zeros_like()
        yg=y.zeros_like()
        gb.SO3part.view(xg).add_DiagCGproduct_back0(gb.SO3part.view(g),gb.SO3part.view(y))
        gb.SO3part.view(yg).add_DiagCGproduct_back1(gb.SO3part.view(g),gb.SO3part.view(x))
        return xg,yg,None


class SO3partArr_GatherFn(torch.autograd.Function): 

    @staticmethod
    def forward(ctx,x,gmap,d):
        ctx.gmap=gmap
        ctx.d=d
        ctx.adims=x.get_adims()
        new_adims=ctx.adims
        new_adims[d]=gmap.d_out()
        r=SO3partArr.zeros(x.getb(),ctx.new_adims,x.getl(),x.getn(),device=x.device)
        r.backend().add_gather(x,gmap,d)
        return r

    @staticmethod
    def backward(ctx,rg):
        xg=SO3partArr.zeros(rg.getb(),ctx.adims,rg.getl(),rg.getn(),device=rg.device)
        xg.backend().add_gather_back(rg,ctx.gmap,ctx.d)
        return xg,None,None


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
        gelib_base.add_conterpolate3dB(_r,_x,_M)
        return r

    @staticmethod
    def backward(ctx,g):
        _g=_ctensor.view(g)
        _M=_rtensor.view(ctx.M)
        gx=SO3partArr.zeros(ctx.b,ctx.adims,ctx.l,ctx.n,g.device)
        _gx=_SO3partB_array.view(gx)
        gelib_base.add_conterpolate3dB_back(_gx,_g,_M)
        return gx,None



## ----------------------------------------------------------------------------------------------------------
## ---- Helpers ---------------------------------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------------------


def CGproduct(x,y,maxl=-1):
    return x.CGproduct(y,maxl)
    


