
# This file is part of GElib, a C++/CUDA library for group
# equivariant tensor operations. 
# 
# Copyright (c) 2022, Imre Risi Kondor
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


import torch
from gelib_base import SO3partB_array as _SO3partB_array


class SO3partArr(torch.Tensor):
    """
    A collection of vectors that transform according to a specific irreducible representation of SO(3).
    The vectors are stacked into a fourth order tensor. The first index is the site index, the second index
    is the batch index, the third is m=-l,...,l, and the fourth index is the fragment index. 
    """

    def __init__(self,_T):
        self=_T


    ## ---- Static constructors -----------------------------------------------------------------------------

    
    @staticmethod
    def zeros(_adims,l,n,_dev=0):
        """
        Create an SO(3)-part consisting of N*b lots of n vectors transforming according to the l'th irrep of SO(3).
        The vectors are initialized to zero, resulting in an b*(2+l+1)*n dimensional complex tensor of zeros.
        """        
        if _dev==0:
            return SO3partArr(torch.zeros(_adims+[2*l+1,n,2]))
        else:
            return SO3partArr(torch.zeros(_adims+[2*l+1,n,2])).cuda()


    @staticmethod
    def randn(_adims,l,n,_dev=0):
        """
        Create an SO(3)-part consisting of N*b lots of n vectors transforming according to the l'th irrep of SO(3).
        The vectors are initialized as random gaussian vectors, resulting in an b*(2+l+1)*n dimensional random
        complex tensor.
        """
        if _dev==0:        
            return SO3partArr(torch.randn(_adims+[2*l+1,n,2]))
        else:
            return SO3partArr(torch.randn(_adims+[2*l+1,n,2],device='cuda'))


    @staticmethod
    def Fzeros(_adims,l,_dev=0):
        """
        Create an SO(3)-part corresponding to the l'th matrix in the Fourier transform of a function on SO(3).
        This gives a N*b*(2+l+1)*(2l+1) dimensional complex tensor. 
        """
        if _dev==0:        
            return SO3partArr(torch.zeros(_adims+[2*l+1,2*l+1,2]))
        else:
            return SO3partArr(torch.zeros(_adims+[2*l+1,2*l+1,2])).cuda()


    @staticmethod
    def Frandn(_adims,l,_dev=0):
        """
        Create an SO(3)-part corresponding to the l'th matrix in the Fourier transform of a function on SO(3).
        This gives a b*(2+l+1)*(2l+1) dimensional complex random tensor. 
        """
        if _dev==0:        
            return SO3partArr(torch.randn(_adims+[2*l+1,2*l+1,2]))
        else:
            return SO3partArr(torch.randn(_adims+[2*l+1,2*l+1,2],device='cuda'))


    ## ---- Access ------------------------------------------------------------------------------------------


    def get_adims(self):
        return list(self.size()[0:self.dim()-3])

    def getl(self):
        return (self.size(-3)-1)/2

    def getn(self):
        return self.size(-2)


    ## ---- Operations --------------------------------------------------------------------------------------


    def rotate(self,R):
        return SO3partArr(_SO3partB_array.view(self).apply(R).torch())


    def gather(self,_mask):
        """
        Gather the elements of this SO3partArr into a new SO3partArr according to the mask
        """
        return SO3partArr(SO3partArr_GatherFn.apply(_mask,self))


    # ---- Products -----------------------------------------------------------------------------------------


    def CGproduct(self, y, l):
        """
        Compute the l component of the Clesbsch--Gordan product of this SO3partArr with another SO3partArr y.
        """
        return SO3partArr_CGproductFn.apply(self,y,l)


    def DiagCGproduct(self, y, l):
        """
        Compute the l component of the diagonal Clesbsch--Gordan product of this SO3partArr with another SO3partArr y.
        """
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

        adims = x.get_adims()
        dev = int(x.is_cuda)
        r = SO3partArr.zeros(adims,l,x.getn()*y.getn(),dev)

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

        adims = x.get_adims()
        dev = int(x.is_cuda)
        r = SO3part.zeros(adims,l,x.getn(),dev)

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
        adims = x.get_adims() #change this
        l=x.getl()
        n=x.getn()
        dev=int(x.is_cuda)
        r=SO3partArr.zeros(adims,l,n,dev)
        
        _x=_SO3partB_array.view(x)
        _r=_SO3partB_array.view(r)
        _r.gather(_x,_mask)

        return r

    @staticmethod
    def backward(ctx,yg):

        #N=yg.size(0)
        #b=yg.size(1)
        #l=int((y.size(2)-1)/2)
        #n=y.size(3)
        #dev=int(y.is_cuda)
        #r=MakeZeroSO3partArrs(N,b,l,n,dev)
        r=torch.zeros_like(yg) # change this

        _x=_SO3partB_array.view(args)
        _r=_SO3partB_array.view(r)
        _r.gather(_x,ctx.mask.inv())

        return tuple([None]+r)




## ----------------------------------------------------------------------------------------------------------
## ---- Helpers ---------------------------------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------------------


def CGproduct(x,y,maxl=-1):
    return x.CGproduct(y,maxl)
    


