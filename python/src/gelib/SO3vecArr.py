
# This file is part of GElib, a C++/CUDA library for group
# equivariant tensor operations. 
# 
# Copyright (c) 2022, Imre Risi Kondor
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


from ctypes import Array
from typing import Iterable, List, Optional
import torch
from gelib_base import SO3partB_array as _SO3partB_array
from gelib_base import SO3vecB_array as _SO3vecB_array

from gelib import SO3weights, SO3weightsArr, SO3partArr
from gelib import DiagCGproductType, DDiagCGproductType


## ----------------------------------------------------------------------------------------------------------
## ---- SO3vecArr -------------------------------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------------------
    

class SO3vecArr:
    """
    An array of SO(3)-covariant vectors consisting of a sequence of SO3part objects, each transforming according
    to a specific irrep of SO(3).
    """
    
    def __init__(self, parts : Optional[Iterable[SO3partArr]] = None):
        if parts is None:
            parts = []

        self.parts = []
        for part in parts:
            self.parts.append(SO3partArr(part))


    ## ---- Static constructors ------------------------------------------------------------------------------


    @staticmethod
    def zeros(b : int, _adims : Optional[List[int]] ,_tau : List[int], device='cpu') -> 'SO3vecArr':
        "Construct a zero SO3vec object of given type _tau."
        if _adims == None:
            _adims = []

        R=SO3vecArr()
        for l in range(0,len(_tau)):
            R.parts.append(SO3partArr.zeros(b, _adims, l, _tau[l], device))
            #R.parts.append(SO3partArr.zeros(_adims,l,_tau[l],_dev))
        return R
    
    @staticmethod
    def ones(b : int, _adims : Optional[List[int]], _tau : List[int], device='cpu') -> 'SO3vecArr':
        "Construct a zero SO3vec object of given type _tau."
        if _adims == None:
            _adims = []

        R=SO3vecArr()
        for l in range(0,len(_tau)):
            R.parts.append(SO3partArr.ones(b, _adims, l, _tau[l], device))
        return R

    @staticmethod
    def randn(b : int, _adims : Optional[List[int]], _tau : List[int], device='cpu') -> 'SO3vecArr':
        "Construct a random SO3vec array of given type _tau."
        if _adims == None:
            _adims = []
            
        R=SO3vecArr()
        for l in range(0,len(_tau)):
            R.parts.append(SO3partArr.randn(b, _adims, l, _tau[l], device))
        return R

    @staticmethod
    def Fzeros(b : int, _adims : Optional[List[int]], maxl : int, device='cpu') -> 'SO3vecArr':
        "Construct an SO3vec array corresponding the to the Forier matrices 0,1,...maxl of b functions on SO(3)."
        if _adims == None:
            _adims = []
            
        R=SO3vecArr()
        for l in range(0,maxl+1):
            R.parts.append(SO3partArr.Fzeros(b, _adims, l, device))
        return R

    @staticmethod
    def Frandn(b : int, _adims : Optional[List[int]], maxl : int, device='cpu') -> 'SO3vecArr':
        "Construct a zero SO3Fvec array with l ranging from 0 to maxl."
        if _adims == None:
            _adims = []
            
        R=SO3vecArr()
        for l in range(0,maxl+1):
            R.parts.append(SO3partArr.Frandn(b, _adims, l, device))
        return R

    @staticmethod
    def zeros_like(x : 'SO3vecArr') -> 'SO3vecArr':
        R=SO3vecArr()
        for l in range(0,len(x.parts)):
            R.parts.append(SO3partArr.zeros_like(x.parts[l]))
        return R
                           
    @staticmethod
    def randn_like(x: 'SO3vecArr') -> 'SO3vecArr':
        R=SO3vecArr()
        for l in range(0,len(x.parts)):
            R.parts.append(SO3partArr.randn_like(x.parts[l]))
        return R
    
    @staticmethod
    def from_part(part : SO3partArr, max_l : int,device='cpu') -> 'SO3vecArr':
        assert isinstance(part, SO3partArr)
        tau = [ part.size()[-1] for _ in range(max_l + 1)]
        result = SO3vecArr.zeros(part.getb(), part.get_adims(), tau, device)
        result.parts[part.getl()] = part
        return result

    @classmethod
    def spharm(self, max_l : int, X : torch.Tensor, device : str = 'cpu') -> 'SO3vecArr':
        """
        Return the spherical harmonics of the vector (x,y,z) for all l <= max_l.
        """
        parts = [ SO3partArr.spharm(l, X, device) for l in range(max_l + 1) ]
        return SO3vecArr(parts)
    
    
    ## ---- Access -------------------------------------------------------------------------------------------


    def getb(self) -> int:
        return self.parts[0].size(0)

    def get_adims(self) -> List[int]:
        assert len(self.parts)>0 
        return list(self.parts[0].size()[1:self.parts[0].dim()-2])

    def get_aadims(self) -> List[int]:
        assert len(self.parts)>0 
        return list(self.parts[0].size()[0:self.parts[0].dim()-2])

    def getN(self) -> int:
        t=1
        for i in range(0,self.parts[0].dim()-2):
            t*=self.parts[0].size(i)
        return t
                       
    def getNarr(self) -> int:
        # TODO: Should this be different than getN()?
        t=1
        for i in range(1,self.parts[0].dim()-2):
            t*=self.parts[0].size(i)
        return t
                       
    def tau(self) -> Array:
        "Return the 'type' of the SO3vec, i.e., how many components it has corresponding to l=0,1,2,..."
        r=[]
        for l in range(0,len(self.parts)):
            r.append(self.parts[l].size(-1))
        return r

    def requires_grad_(self):
        for p in self.parts:
            p.requires_grad_()

    def get_grad(self):
        r = SO3vecArr()
        for p in self.parts:
            r.parts.append(p.grad)
        return r
    
    def getPart(self, l_val : int) -> SO3partArr:
        assert 0 <= l_val and l_val < len(self.parts)
        return SO3partArr(self.parts[l_val])
    
    def getLMax(self) -> int:
        return len(self.parts) - 1


    ## ---- Transport ---------------------------------------------------------------------------------------


    def to(self,device) -> 'SO3vecArr':
        r=SO3vecArr()
        for p in self.parts:
            r.parts.append(p.to(device))
        return r


    ## ---- Operations ---------------------------------------------------------------------------------------


    def rotate(self, R) -> 'SO3vecArr':
        "Apply the group element to this vector"
        r=SO3vecArr()
        for l in range(0,len(self.parts)):
            r.parts.append(_SO3partB_array.view(self.parts[l]).rotate(R).torch())
        return r

    def __add__(self, y : 'SO3vecArr') -> 'SO3vecArr':
        assert isinstance(y, SO3vecArr)
        if(len(self.parts)!=len(y.parts)):
            raise IndexError("SO3vecArr must have the same number of parts.")
        R=SO3vecArr()
        for l in range(len(self.parts)):
            R.parts.append(self.parts[l]+y.parts[l])
        return R

    def __sub__(self, y : 'SO3vecArr') -> 'SO3vecArr':
        assert isinstance(y, SO3vecArr)
        if(len(self.parts)!=len(y.parts)):
            raise IndexError("SO3vecArr must have the same number of parts.")
        R=SO3vecArr()
        for l in range(len(self.parts)):
            R.parts.append(self.parts[l] + (-1) * y.parts[l])
        return R

    def __mul__(self, w : 'SO3vecArr' | SO3weightsArr | SO3weights) -> 'SO3vecArr':
        if(isinstance(w,SO3weights)):
            if(len(self.parts)!=len(w.parts)):
                raise IndexError("SO3vecArr and SO3weights have different number of parts.")
            R=SO3vecArr()
            aadims=self.get_aadims()
            N=self.getN()
            for l in range(len(self.parts)):
                n=self.parts[l].size(-1)
                m=w.parts[l].size(1)
                x=self.parts[l].reshape([N*(2*l+1),n])
                R.parts.append(torch.matmul(x,w.parts[l]).reshape(aadims+[2*l+1,m]))
            return R
        if(isinstance(w,SO3weightsArr)):
            if(len(self.parts)!=len(w.parts)):
                raise IndexError("SO3vecArr and SO3weightsArr have different number of parts.")
            R=SO3vecArr()
            adims=self.get_adims()
            assert(w.get_adims()==adims)
            N=self.getNarr()
            for l in range(len(self.parts)):
                b=self.getb()
                n=w.parts[l].size(-2)
                m=w.parts[l].size(-1)
                x=self.parts[l].reshape([b,N,(2*l+1),n])
                t=torch.einsum('...ami,...aij->...amj',x,w.parts[l].reshape([N,n,m]))
                R.parts.append(t.reshape([b]+adims+[2*l+1,m]))
            return R
        if(isinstance(w,torch.Tensor)):
            R=SO3vecArr()
            for l in range(len(self.parts)):
                R.parts.append(torch.matmul(self.parts[l],w))
            return R
        raise TypeError("SO3vecArr can only be multiplied by a scalar tensor, an SO3weights or an SOweightsArr.")

    def mix1(self, w : SO3weights) -> 'SO3vecArr':
        assert isinstance(w,SO3weights)
        if(len(self.parts)!=len(w.parts)):
            raise IndexError("SO3vecArr and SO3weights have different number of parts.")
        R=SO3vecArr()
        for l in range(len(self.parts)):
            R.parts.append(torch.einsum('...imk,...ij->...jmk',self.parts[l],w.parts[l]))
        return R
    
    def odot(self,
             y : 'SO3vecArr',
             last_dim : Optional[int] = None) -> float | torch.Tensor:
        assert isinstance(y, SO3vecArr)
        assert(len(self.parts)==len(y.parts))

        # Simple case where the result is a number.
        if last_dim is None:
            r=0
            for l in range(0, len(self.parts)):
                self_parts = torch.view_as_real(self.parts[l])
                y_parts = torch.view_as_real(y.parts[l])
                mul = torch.mul(self_parts, y_parts)
                r+=torch.sum(mul)
            return r
        
        # Handle negative indices.
        if last_dim < 0:
            last_dim += self.parts[0].dim()
            assert last_dim >= 0
        assert last_dim < self.parts[0].dim() - 2

        # Create the starting "zero".
        size = list(self.parts(0).size())[:last_dim + 1]
        size.append(size[-1])
        r = torch.zeros(size,
                        device = self.parts[0].device,
                        requires_grad = False)
        
        # Extra indices for einsum().
        extra_letters = "cdefghmnopqrstuvwxyz"

        # Sum across all rotation orders.
        for l in range(0, len(self.parts)):
            # TODO: Do I need view_as_real()?
            self_parts = self.parts[l]
            y_parts = y.parts[l]
            
            # Extra 1 for the new dimension from view_as_real().
            summed_dims = self_parts.dim() - last_dim # -1
            assert summed_dims < len(extra_letters)

            # entry []...,i,j] is the sum of self[...,i,...]*other[...,j,...].
            ein = "...a{0},...b{0}->...ab".format(extra_letters[:summed_dims])
            einsum = torch.einsum(ein, self_parts, y_parts)
            assert einsum.size() == r.size()
            r = r + einsum
        return r

    def gather(self,_mask):
        """
        Gather the elements of this SO3vecArr into a new SO3vecArr according to the mask
        """
        r=SO3vecArr()
        r.parts=list(SO3vecArr_GatherFn.apply(_mask,*(self.parts)))
        return r
        
        
    ## ---- Products -----------------------------------------------------------------------------------------


    def CGproduct(self, y : 'SO3vecArr', maxl : Optional[int] =-1) -> 'SO3vecArr':
        """
        Compute the full Clesbsch--Gordan product of this SO3vecArr with another SO3vecArr y.
        """
        assert isinstance(y, SO3vecArr)
        if maxl == None:
            maxl = -1
            
        r=SO3vecArr()
        r.parts=list(SO3vecArr_CGproductFn.apply(len(self.parts),len(y.parts),maxl,*(self.parts+y.parts)))
        return r


    def ReducingCGproduct(self, y : 'SO3vecArr', maxl : Optional[int] =-1) -> 'SO3vecArr':
        """
        Compute the full Clesbsch--Gordan product of this SO3vecArr with another SO3vecArr y,
        reducing over extra array dimensions as necesssary.
        """
        assert isinstance(y, SO3vecArr)
        if maxl == None:
            maxl = -1

        r=SO3vecArr()
        r.parts=list(SO3vecArr_ReducingCGproductFn.apply(len(self.parts),len(y.parts),maxl,*(self.parts+y.parts)))
        return r


    def DiagCGproduct(self, y : 'SO3vecArr', maxl : Optional[int] =-1) -> 'SO3vecArr':
        """
        Compute the diagonal Clesbsch--Gordan product of this SO3vecArr with another SO3vecArr y.
        """
        assert isinstance(y, SO3vecArr)
        if maxl == None:
            maxl = -1

        r=SO3vecArr()
        r.parts=list(SO3vecArr_DiagCGproductFn.apply(len(self.parts),len(y.parts),maxl,*(self.parts+y.parts)))
        return r


    def DiagCGproductB(self, y : 'SO3vecArr', maxl : Optional[int] =-1) -> 'SO3vecArr':
        """
        Compute the diagonal Clesbsch--Gordan product of this SO3vecArr with another SO3vecArr y.
        """
        assert isinstance(y, SO3vecArr)
        r=SO3vecArr()
        r.parts=list(SO3vecArr_DiagCGproductBFn.apply(len(self.parts),len(y.parts),maxl,*(self.parts+y.parts)))
        return r


    def DDiagCGproduct(self, y : 'SO3vecArr', maxl : Optional[int] =-1) -> 'SO3vecArr':
        """
        Compute the reduced diagonal Clesbsch--Gordan product of this SO3vecArr with another SO3vecArr y.
        """
        assert isinstance(y, SO3vecArr)
        r=SO3vecArr()
        r.parts=list(SO3vecArr_DDiagCGproductFn.apply(len(self.parts),len(y.parts),maxl,*(self.parts+y.parts)))
        return r


    def Fproduct(self, y : 'SO3vecArr', maxl : Optional[int] =-1) -> 'SO3vecArr':
        """
        Compute the Fourier space product of this SO3Fvec with another SO3Fvec y.
        """
        assert isinstance(y, SO3vecArr)
        r=SO3vecArr()
        r.parts=list(SO3vecArr_FproductFn.apply(len(self.parts),len(y.parts),maxl,*(self.parts+y.parts)))
        return r


    def Fmodsq(self, maxl : Optional[int] =-1) -> 'SO3vecArr':
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


def CGproduct(x : SO3vecArr, y : SO3vecArr, maxl : Optional[int] = -1) -> SO3vecArr:
    assert isinstance(x, SO3vecArr)
    return x.CGproduct(y,maxl)
    
def ReducingCGproduct(x : SO3vecArr, y : SO3vecArr, maxl : Optional[int] = -1) -> SO3vecArr:
    assert isinstance(x, SO3vecArr)
    return x.ReducingCGproduct(y,maxl)
    
def DiagCGproduct(x : SO3vecArr, y : SO3vecArr, maxl : Optional[int] = -1) -> SO3vecArr:
    assert isinstance(x, SO3vecArr)
    return x.DiagCGproduct(y,maxl)
    
def DiagCGproductB(x : SO3vecArr, y : SO3vecArr, maxl : Optional[int] = -1) -> SO3vecArr:
    assert isinstance(x, SO3vecArr)
    return x.DiagCGproductB(y,maxl)
    
def DDiagCGproduct(x : SO3vecArr, y : SO3vecArr, maxl : Optional[int] = -1) -> SO3vecArr:
    assert isinstance(x, SO3vecArr)
    return x.DDiagCGproduct(y,maxl)
    
def Fproduct(x : SO3vecArr, y : SO3vecArr, maxl : Optional[int] = -1) -> SO3vecArr:
    assert isinstance(x, SO3vecArr)
    return x.Fproduct(y,maxl)

def Fmodsq(x : SO3vecArr, a : Optional[int] = -1) -> SO3vecArr:
    assert isinstance(x, SO3vecArr)
    return x.Fmodsq(a)


def tau_type(x : int) -> List[int]:
    r=[]
    for t in x:
        r.append(t.size(-1))
    return r

def CGproductType(x, y, maxl : Optional[int] = -1):
    if maxl==-1:
        maxl=len(x)+len(y)-2
    r=[0]*(maxl+1)
    for l1 in range(0,len(x)):
        for l2 in range(0,len(y)):
            for l in range(abs(l1-l2),min(l1+l2,maxl)+1):
                r[l]+=x[l1]*y[l2]
    return r


def MakeZeroSO3partArrs(badims,tau,device):
    R=[]
    for l in range(0,len(tau)):
        R.append(torch.zeros(badims+[2*l+1,tau[l]],dtype=torch.cfloat,device=device))
    return R


def makeZeroSO3FpartArrs(badims,maxl,device):
    R=[]
    for l in range(0,maxl+1):
        R.append(torch.zeros(badims+[2*l+1,2*l+1],dtype=torch.cfloat,device=device))
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

        adims=list(args[0].size()[0:args[0].dim()-2])
        tau=CGproductType(tau_type(args[0:k1]),tau_type(args[k1:k1+k2]),maxl)
        r=MakeZeroSO3partArrs(adims,tau,args[0].device)

        _x=_SO3vecB_array.view(args[0:k1])
        _y=_SO3vecB_array.view(args[k1:k1+k2])
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

        _x=_SO3vecB_array.view(inputs[0:k1])
        _y=_SO3vecB_array.view(inputs[k1:k1+k2])

        _g=_SO3vecB_array.view(args)
        _xg=_SO3vecB_array.view(grads[3:k1+3])
        _yg=_SO3vecB_array.view(grads[k1+3:k1+k2+3])

        _xg.addCGproduct_back0(_g,_y)
        _yg.addCGproduct_back1(_g,_x)

        return tuple(grads)


class SO3vecArr_ReducingCGproductFn(torch.autograd.Function): 

    @staticmethod
    def forward(ctx,k1,k2,maxl,*args):
        ctx.k1=k1
        ctx.k2=k2
        #ctx.maxl=maxl
        ctx.save_for_backward(*args)

        adimsx=list(args[0].size()[0:args[0].dim()-2])
        adimsy=list(args[k1].size()[0:args[0].dim()-2])
        if len(adimsx)<len(adimsy):
            adims=adimsx
        else:
            adims=adimsy
        tau=CGproductType(tau_type(args[0:k1]),tau_type(args[k1:k1+k2]),maxl)
        r=MakeZeroSO3partArrs(adims,tau,args[0].device)

        _x=_SO3vecB_array.view(args[0:k1])
        _y=_SO3vecB_array.view(args[k1:k1+k2])
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

        _x=_SO3vecB_array.view(inputs[0:k1])
        _y=_SO3vecB_array.view(inputs[k1:k1+k2])

        _g=_SO3vecB_array.view(args)
        _xg=_SO3vecB_array.view(grads[3:k1+3])
        _yg=_SO3vecB_array.view(grads[k1+3:k1+k2+3])

        _xg.addCGproduct_back0(_g,_y)
        _yg.addCGproduct_back1(_g,_x)

        return tuple(grads)


class SO3vecArr_DiagCGproductFn(torch.autograd.Function): 

    @staticmethod
    def forward(ctx,k1,k2,maxl,*args):
        ctx.k1=k1
        ctx.k2=k2
        #ctx.maxl=maxl
        ctx.save_for_backward(*args)

        adims=list(args[0].size()[0:args[0].dim()-2])
        tau=DiagCGproductType(tau_type(args[0:k1]),tau_type(args[k1:k1+k2]),maxl)
        r=MakeZeroSO3partArrs(adims,tau,args[0].device)

        _x=_SO3vecB_array.view(args[0:k1])
        _y=_SO3vecB_array.view(args[k1:k1+k2])
        _r=_SO3vecB_array.view(r)
        _r.addDiagCGproduct(_x,_y)

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

        _x=_SO3vecB_array.view(inputs[0:k1])
        _y=_SO3vecB_array.view(inputs[k1:k1+k2])

        _g=_SO3vecB_array.view(args)
        _xg=_SO3vecB_array.view(grads[3:k1+3])
        _yg=_SO3vecB_array.view(grads[k1+3:k1+k2+3])

        _xg.addDiagCGproduct_back0(_g,_y)
        _yg.addDiagCGproduct_back1(_g,_x)

        return tuple(grads)


class SO3vecArr_DiagCGproductBFn(torch.autograd.Function): 

    @staticmethod
    def forward(ctx,k1,k2,maxl,*args):
        ctx.k1=k1
        ctx.k2=k2
        #ctx.maxl=maxl
        ctx.save_for_backward(*args)

        adims=list(args[0].size()[0:args[0].dim()-2])
        tau=DiagCGproductType(tau_type(args[0:k1]),tau_type(args[k1:k1+k2]),maxl)
        r=MakeZeroSO3partArrs(adims,tau,args[0].device)

        _x=_SO3vecB_array.view(args[0:k1])
        _y=_SO3vecB_array.view(args[k1:k1+k2])
        _r=_SO3vecB_array.view(r)
        _r.addDiagCGproductB(_x,_y)

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

        _x=_SO3vecB_array.view(inputs[0:k1])
        _y=_SO3vecB_array.view(inputs[k1:k1+k2])

        _g=_SO3vecB_array.view(args)
        _xg=_SO3vecB_array.view(grads[3:k1+3])
        _yg=_SO3vecB_array.view(grads[k1+3:k1+k2+3])

        _xg.addDiagCGproductB_back0(_g,_y)
        _yg.addDiagCGproductB_back1(_g,_x)

        return tuple(grads)


class SO3vecArr_DDiagCGproductFn(torch.autograd.Function): 

    @staticmethod
    def forward(ctx,k1,k2,maxl,*args):
        ctx.k1=k1
        ctx.k2=k2
        ctx.save_for_backward(*args)

        adims=list(args[0].size()[0:args[0].dim()-2])
        tau=DDiagCGproductType(tau_type(args[0:k1]),maxl)
        r=MakeZeroSO3partArrs(adims,tau,args[0].device)
        
        _x=_SO3vecB_array.view(args[0:k1])
        _y=_SO3vecB_array.view(args[k1:k1+k2])
        _r=_SO3vecB_array.view(r)
        _r.addDDiagCGproduct(_x,_y)

        return tuple(r)

    @staticmethod
    def backward(ctx,*args):

        k1=ctx.k1
        k2=ctx.k2
        inputs=ctx.saved_tensors
        assert len(inputs)==k1+k2, "Wrong number of saved tensors."

        grads=[None,None,None]
        for i in range(k1+k2):
            grads.append(torch.zeros_like(inputs[i]))

        _x=_SO3vecB_array.view(inputs[0:k1])
        _y=_SO3vecB_array.view(inputs[k1:k1+k2])

        _g=_SO3vecB_array.view(args)
        _xg=_SO3vecB_array.view(grads[3:k1+3])
        _yg=_SO3vecB_array.view(grads[k1+3:k1+k2+3])

        _xg.addDDiagCGproduct_back0(_g,_y)
        _yg.addDDiagCGproduct_back1(_g,_x)

        return tuple(grads)


class SO3vecArr_FproductFn(torch.autograd.Function): #todo

    @staticmethod
    def forward(ctx,k1,k2,_maxl,*args):
        ctx.k1=k1
        ctx.k2=k2
        ctx.save_for_backward(*args)

        adims=list(args[0].size()[0:args[0].dim()-2])
        if _maxl==-1:
            maxl=k1+k2-2
        else:
            maxl=_maxl

        r=makeZeroSO3FpartArrs(adims,maxl,args[0].device)

        _x=_SO3vecB_array.view(args[0:k1])
        _y=_SO3vecB_array.view(args[k1:k1+k2])
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

        _x=_SO3vecB_array.view(inputs[0:k1])
        _y=_SO3vecB_array.view(inputs[k1:k1+k2])

        _g=_SO3vecB_array.view(args)
        _xg=_SO3vecB_array.view(grads[3:k1+3])
        _yg=_SO3vecB_array.view(grads[k1+3:k1+k2+3])

        _xg.addFproduct_back0(_g,_y)
        _yg.addFproduct_back1(_g,_x)

        return tuple(grads)


class SO3vecArr_FmodsqFn(torch.autograd.Function): #todo

    @staticmethod
    def forward(ctx,k1,_maxl,*args):
        ctx.k1=k1
        #ctx.k2=k1
        ctx.save_for_backward(*args)

        adims=list(args[0].size()[0:args[0].dim()-2])
        if _maxl==-1:
            maxl=k1+k1-2
        else:
            maxl=_maxl

        r=makeZeroSO3FpartArrs(adims,maxl,args[0].device)

        _x=_SO3vecB_array.view(args[0:k1])
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

        _x=_SO3vecB_array.view(inputs[0:k1])

        _g=_SO3vecB_array.view(args)
        _xg=_SO3vecB_array.view(grads[2:k1+2])

        _xg.addFproduct_back0(_g,_x)
        _xg.addFproduct_back1(_g,_x)

        return tuple(grads)


class SO3vecArr_GatherFn(torch.autograd.Function): 

    @staticmethod
    def forward(ctx,*args):

        ctx.mask=args[0]
        ctx.adims=list(args[1].size()[0:args[1].dim()-2])
        tau=tau_type(args[1:])
        #dev=int(args[1].is_cuda)
        r=MakeZeroSO3partArrs(ctx.adims,tau,args[1].device)
        
        _x=_SO3vecB_array.view(args[1:])
        _r=_SO3vecB_array.view(r)
        _r.add_gather(_x,args[0])

        return tuple(r)

    @staticmethod
    def backward(ctx,*args):

        tau=tau_type(args)
        dev=args[0].device
        r=MakeZeroSO3partArrs(ctx.adims,tau,dev)

        _x=_SO3vecB_array.view(args)
        _r=_SO3vecB_array.view(r)
        _r.add_gather(_x,ctx.mask.inv())

        return tuple([None]+r)


# ---- Helpers -----------------------------------------------------------------------------------------------


