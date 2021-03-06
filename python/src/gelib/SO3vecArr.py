
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
from gelib_base import SO3vecB_array as _SO3vecB_array

from gelib import *


## ----------------------------------------------------------------------------------------------------------
## ---- SO3vecArr -------------------------------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------------------
    

class SO3vecArr:
    """
    An array of SO(3)-covariant vectors consisting of a sequence of SO3part objects, each transforming according
    to a specific irrep of SO(3).
    """
    
    def __init__(self):
        self.parts=[]


    ## ---- Static constructors ------------------------------------------------------------------------------


    @staticmethod
    def zeros(_adims,_tau,_dev=0):
        "Construct a zero SO3vec object of given type _tau."
        R=SO3vecArr()
        for l in range(0,len(_tau)):
            R.parts.append(SO3partArr.zeros(_adims,l,_tau[l],_dev))
        return R

    @staticmethod
    def randn(_adims,_tau,_dev=0):
        "Construct a random SO3vec array of given type _tau."
        R=SO3vecArr()
        for l in range(0,len(_tau)):
            R.parts.append(SO3partArr.randn(_adims,l,_tau[l],_dev))
        return R

    @staticmethod
    def Fzeros(_adims,maxl,_dev=0):
        "Construct an SO3vec array corresponding the to the Forier matrices 0,1,...maxl of b functions on SO(3)."
        R=SO3vecArr()
        for l in range(0,maxl+1):
            R.parts.append(SO3partArr.Fzeros(_adims,l,_dev))
        return R

    @staticmethod
    def Frandn(_adims,maxl,_dev=0):
        "Construct a zero SO3Fvec array with l ranging from 0 to maxl."
        R=SO3vecArr()
        for l in range(0,maxl+1):
            R.parts.append(SO3partArr.Frandn(_adims,l,_dev))            
        return R

    @staticmethod
    def zeros_like(x):
        R=SO3vecrArr()
        # b=x.parts[0].dim(0)
        for l in range(0,len(x.parts)):
            R.parts.append(SO3partArr(torch.zeros_like(x.parts[l])))
        return R;
                           
                       
    
    ## ---- Access -------------------------------------------------------------------------------------------


    def get_adims(self):
        assert len(parts)>0 
        return parts[0].get_adims()

    def tau(self):
        "Return the 'type' of the SO3vec, i.e., how many components it has corresponding to l=0,1,2,..."
        r=[]
        for l in range(0,len(self.parts)):
            r.append(self.parts[l].getn())
        return r


    ## ---- Transport ---------------------------------------------------------------------------------------


    def to(self,device):
        r=SO3vecArr()
        for p in self.parts:
            r.parts.append(p.to(device))
        return r


    ## ---- Operations ---------------------------------------------------------------------------------------


    def rotate(self,R):
        "Apply the group element to this vector"
        r=SO3vecArr()
        for l in range(0,len(self.parts)):
            r.parts.append(self.parts[l].rotate(R))
        return r


    def gather(self,_mask):
        """
        Gather the elements of this SO3vecArr into a new SO3vecArr according to the mask
        """
        r=SO3vecArr()
        r.parts=list(SO3vecArr_GatherFn.apply(_mask,*(self.parts)))
        return r
        
        
    ## ---- Products -----------------------------------------------------------------------------------------


    def CGproduct(self,y,maxl=-1):
        """
        Compute the full Clesbsch--Gordan product of this SO3vecArr with another SO3vecArr y.
        """
        r=SO3vecArr()
        r.parts=list(SO3vecArr_CGproductFn.apply(len(self.parts),len(y.parts),maxl,*(self.parts+y.parts)))
        return r


    def Fproduct(self,y,maxl=-1):
        """
        Compute the Fourier space product of this SO3Fvec with another SO3Fvec y.
        """
        r=SO3vecArr()
        r.parts=list(SO3vecArr_FproductFn.apply(len(self.parts),len(y.parts),maxl,*(self.parts+y.parts)))
        return r


    def Fmodsq(self,maxl=-1):
        """
        Compute the Fourier transform of the squared modulus of f. 
        """
        r=SO3vecArr()
        r.parts=list(SO3vecArr_FmodsqFn.apply(len(self.parts),maxl,*(self.parts)))
        return r


    ## ---- I/O ----------------------------------------------------------------------------------------------

        
    def __repr__(self):
        u=_SO3vecB_array.view(self.parts)
        return u.__repr__()

    def __str__(self):
        u=_SO3vecB_array.view(self.parts)
        return u.__str__()



## ----------------------------------------------------------------------------------------------------------
## ---- Other functions --------------------------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------------------


def CGproduct(x,y,maxl=-1):
    return x.CGproduct(y,maxl)
    
def Fproduct(x,y,a=-1):
    return x.Fproduct(y,a)

def Fmodsq(x,a=-1):
    return x.Fmodsq(a)


def tau_type(x):
    r=[]
    for t in x:
        r.append(t.getn())
    return r

def CGproductType(x,y,maxl=-1):
    if maxl==-1:
        maxl=len(x)+len(y)-2
    r=[0]*(maxl+1)
    for l1 in range(0,len(x)):
        for l2 in range(0,len(y)):
            for l in range(abs(l1-l2),min(l1+l2,maxl)+1):
                r[l]+=x[l1]*y[l2]
    return r


def MakeZeroSO3partArrs(adims,tau,_dev=0):
    R=[]
    for l in range(0,len(tau)):
        R.append(SO3partArr.zeros(adims,l,tau[l],_dev))
    return R


def makeZeroSO3FpartArrs(adims,maxl,_dev=0):
    R=[]
    for l in range(0,maxl+1):
        R.append(SO3part.Fzeros(adims,l,_dev))
    return R



## ----------------------------------------------------------------------------------------------------------
## ---- Autograd functions -----------------------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------------------


class SO3vecArr_CGproductFn(torch.autograd.Function): 

    @staticmethod
    def forward(ctx,k1,k2,maxl,*args):
        ctx.k1=k1
        ctx.k2=k2
        #ctx.maxl=maxl
        ctx.save_for_backward(*args)

        adims=list(args[0].size()[0:args[0].dim()-3])
        tau=CGproductType(tau_type(args[0:k1]),tau_type(args[k1:k1+k2]),maxl)
        dev=int(args[0].is_cuda)
        r=MakeZeroSO3partArrs(adims,tau,dev)

        _x=_SO3vecB_array.view(args[0:k1]);
        _y=_SO3vecB_array.view(args[k1:k1+k2]);
        _r=_SO3vecB_array.view(r)
        _r.addCGproduct(_x,_y)

        return tuple(r)

    @staticmethod
    def backward(ctx,*args):

        k1=ctx.k1
        k2=ctx.k2
        #maxl=ctx.maxl

        inputs=ctx.saved_tensors
        assert len(inputs)==k1+k2, "Wrong number of saved tensors."

        grads=[None,None,None]
        for i in range(k1+k2):
            grads.append(torch.zeros_like(inputs[i]))

        _x=_SO3vecB_array.view(inputs[0:k1]);
        _y=_SO3vecB_array.view(inputs[k1:k1+k2]);

        _g=_SO3vecB_array.view(args);
        _xg=_SO3vecB_array.view(grads[3:k1+3]);
        _yg=_SO3vecB_array.view(grads[k1+3:k1+k2+3]);

        _xg.addCGproduct_back0(_g,_y)
        _yg.addCGproduct_back1(_g,_x)

        return tuple(grads)


class SO3vec_FproductFn(torch.autograd.Function): #todo

    @staticmethod
    def forward(ctx,k1,k2,_maxl,*args):
        ctx.k1=k1
        ctx.k2=k2
        ctx.save_for_backward(*args)

        adims=list(args[0].size()[0:args[0].dim()-3])
        if _maxl==-1:
            maxl=k1+k2-2
        else:
            maxl=_maxl
        dev=int(args[0].is_cuda)

        r=makeZeroSO3Fparts(adims,maxl,dev)

        _x=_SO3vecB_array.view(args[0:k1]);
        _y=_SO3vecB_array.view(args[k1:k1+k2]);
        _r=_SO3vecB_array.view(r)
        _r.addFproduct(_x,_y)

        return tuple(r)

    @staticmethod
    def backward(ctx,*args):

        k1=ctx.k1
        k2=ctx.k2
        #maxl=ctx.maxl

        inputs=ctx.saved_tensors
        assert len(inputs)==k1+k2, "Wrong number of saved tensors."

        grads=[None,None,None]
        for i in range(k1+k2):
            grads.append(torch.zeros_like(inputs[i]))

        _x=_SO3vecB_array.view(inputs[0:k1]);
        _y=_SO3vecB_array.view(inputs[k1:k1+k2]);

        _g=_SO3vecB_array.view(args);
        _xg=_SO3vecB_array.view(grads[3:k1+3]);
        _yg=_SO3vecB_array.view(grads[k1+3:k1+k2+3]);

        _xg.addFproduct_back0(_g,_y)
        _yg.addFproduct_back1(_g,_x)

        return tuple(grads)


class SO3vec_FmodsqFn(torch.autograd.Function): #todo

    @staticmethod
    def forward(ctx,k1,_maxl,*args):
        ctx.k1=k1
        #ctx.k2=k1
        ctx.save_for_backward(*args)

        adims=list(args[0].size()[0:args[0].dim()-3])
        if _maxl==-1:
            maxl=k1+k1-2
        else:
            maxl=_maxl
        dev=int(args[0].is_cuda)

        r=makeZeroSO3Fparts(adims,maxl,dev)

        _x=_SO3vecB_array.view(args[0:k1]);
        #_y=_SO3vecB_array.view(args[k1:k1+k2]);
        _r=_SO3vecB_array.view(r)
        _r.addFproduct(_x,_x)

        return tuple(r)

    @staticmethod
    def backward(ctx,*args):

        k1=ctx.k1
        #k2=ctx.k2

        inputs=ctx.saved_tensors
        assert len(inputs)==k1, "Wrong number of saved tensors."

        grads=[None,None]
        for i in range(k1):
            grads.append(torch.zeros_like(inputs[i]))

        _x=_SO3vecB_array.view(inputs[0:k1]);
        #_y=_SO3vecB_array.view(inputs[k1:k1+k2]);

        _g=_SO3vecB_array.view(args);
        _xg=_SO3vecB_array.view(grads[2:k1+2]);
        #_yg=_SO3vecB_array.view(grads[k1+3:k1+k2+3]);

        _xg.addFproduct_back0(_g,_x)
        _xg.addFproduct_back1(_g,_x)

        return tuple(grads)


class SO3vecArr_GatherFn(torch.autograd.Function): 

    @staticmethod
    def forward(ctx,*args):

        ctx.mask=args[0]
        ctx.adims=list(args[1].size()[0:args[1].dim()-3])
        tau=tau_type(args[1:])
        dev=int(args[1].is_cuda)
        r=MakeZeroSO3partArrs(ctx.adims,tau,dev)
        
        _x=_SO3vecB_array.view(args[1:])
        _r=_SO3vecB_array.view(r)
        _r.gather(_x,args[0])

        return tuple(r)

    @staticmethod
    def backward(ctx,*args):

        tau=tau_type(args)
        dev=int(args[0].is_cuda)
        r=MakeZeroSO3partArrs(ctx.adims,tau,dev)

        _x=_SO3vecB_array.view(args)
        _r=_SO3vecB_array.view(r)
        _r.gather(_x,ctx.mask.inv())

        return tuple([None]+r)


