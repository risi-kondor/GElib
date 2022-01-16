import torch

from gelib_base import SO3part as _SO3part
from gelib_base import SO3vec as _SO3vec

from ctens import *


## ---- SO3part --------------------------------------------------------------------------------------------


class SO3part(ctens):
    """
    A collection of vectors that transform according to a specific irreducible representation of SO(3).
    The parts are stacked into a complex matrix of type ctens. 
    """
    
    def __init__(self,_T):
        self=_T
        self.bsize=_T.bsize # doesn't work
        

    ## ---- Static constructors -----------------------------------------------------------------------------

        
    @staticmethod
    def zeros(l,n):
        """
        Create an SO(3)-part consisting of n vectors transforming according to the l'th irrep of SO(3).
        The vectors are initialized to zero, resulting in an (2+l+1)*n dimensional complex matrix of zeros
        stored as a ctens object.
        """        
        return SO3part(ctens.zeros([2*l+1,n]))

    @staticmethod
    def randn(l,n):
        """
        Create an SO(3)-part consisting of n vectors transforming according to the l'th irrep of SO(3).
        The vectors are initialized as random gaussian vectors, resulting in an (2+l+1)*n dimensional
        random complex matrix stored as a ctens object.
        """        
        return SO3part(ctens.randn([2*l+1,n]))


    @staticmethod
    def Fzeros(l,n):
        """
        Create an SO(3)-part consisting of n blocks of (2l+1) vectors each transforming according to the
        l'th irrep of SO(3). The vectors are initialized to zero, giving a (2+l+1)*(2l+1)*n dimensional
        complex matrix of zeros stored as a ctens object. 
        """        
        r=SO3part(ctens.zeros([2*l+1,(2*l+1)*n],2*l+1))
        r.bsize=2*l+1 # why need this??
        return r

    @staticmethod
    def Frandn(l,n):
        """
        Create an SO(3)-part consisting of n blocks of (2l+1) vectors each transforming according to the
        l'th irrep of SO(3). The vectors are initialized as random Gaussian vectors, giving a
        (2+l+1)*(2l+1)*n dimensional random complex matrix of stored as a ctens object.
        """        
        r= SO3part(ctens.randn([2*l+1,(2*l+1)*n],2*l+1))
        r.bsize=2*l+1 # why need this??
        return r


    ## ---- Access ------------------------------------------------------------------------------------------


    def getl(self):
        "Return l, the index of the representation."
        return int((self.size(1)-1)/2)

    def getn(self):
        "Return n, the number of vectors."
        return self.size(2)


    ## ---- Operations ---------------------------------------------------------------------------------------


    def __mul__(self,W):
        """
        Multiply this SO3part with a  matrix W from the right. There is no distincition between columns
        belonging to the same block or not.
        """
        return SO3part(self.matmul(y))

    def mix(self,W):
        """
        Multiply this SO3part with a weight matrix along the channel dimension, i.e., mix the blocks
        of the SO3part with W.
        """
        return SO3part(self.mix_blocks(2,W))

    def convolve(self,y):
        "Convolve this SO3part with another SO3part. Each block will be convolved with the same block of y."
        return SO3Fpart(self.matmul_each_block_by_corresponding_block(y))

    def FullCGproduct(self,y,l,offs=0):
        "Compute the CG product of each vector in x with each vector in y."
        assert l>0, "Output l must be specified."
        assert abs(self.getl()-y.getl())<=l<=(self.getl()+y.getl()), "l out of range"
        return SO3part_FullCGproductFn.apply(self,y,l,offs)

    def BlockwiseCGproduct(self,y,l,offs=0):
        "Compute the CG product of each vector in x in a given block with each vector in y in the same block."
        assert l>0, "Output l must be specified."
        assert abs(self.getl()-y.getl())<=l<=(self.getl()+y.getl()), "l out of range"
        return SO3part_BlockwiseCGproductFn.apply(self,y,l,offs)

    def DiagCGproduct(self,y,l,offs=0):
        "Compute the CG product of each vector in x with the corresponding vector in y."
        assert l>0, "Output l must be specified."
        assert abs(self.getl()-y.getl())<=l<=(self.getl()+y.getl()), "l out of range"
        return SO3part_DiagCGproductFn.apply(self,y,l,offs)


    ## ---- I/O ----------------------------------------------------------------------------------------------


    def __str__(self):
        return super().__str__()

    def __repr__(self):
        return "<SO3part("+str(self.getl())+","+str(self.getn())+")>"



## ----------------------------------------------------------------------------------------------------------
## ---- SO3vec ----------------------------------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------------------
    

class SO3vec:
    """
    An SO(3)-covariant vector consisting of a sequence of SO3part objects, each transforming according
    to a specific irrep of SO(3).
    """
    
    def __init__(self):
        self.parts=[]


    ## ---- Static constructors ------------------------------------------------------------------------------


    @staticmethod
    def zeros(_tau):
        "Construct a zero SO3vec object of given type _tau."
        R=SO3vec()
        for l in range(0,len(_tau)):
            R.parts.append(SO3part.zeros(l,_tau[l]))
        return R

    @staticmethod
    def randn(_tau):
        "Construct a random SO3vec object of given type _tau."
        R=SO3vec()
        for l in range(0,len(_tau)):
            R.parts.append(SO3part.randn(l,_tau[l]))
        return R


    @staticmethod
    def Fzeros(_maxl,_n):
        "Construct a zero SO3vec object with parts l=0,1,...,maxl corresponding to the union of _n Fourier transforms."
        R=SO3vec()
        for l in range(0,_maxl+1):
            R.parts.append(SO3part.Fzeros(l,_n))
        return R

    @staticmethod
    def Frandn(_maxl,_n):
        "Construct a random SO3vec object with parts l=0,1,...,maxl corresponding to the union of _n Fourier transforms."
        R=SO3vec()
        for l in range(0,_maxl+1):
            R.parts.append(SO3part.Frandn(l,_n))
        return R


    ## ---- Access -----------------------------------------------------------------------------------------


    def tau(self):
        "Return the 'type' of the SO3vec, i.e., how many components it has corresponding to l=0,1,2,..."
        r=[]
        for l in range(0,len(self.parts)):
            r.append(self.parts[l].getn())
        return r


    ## ---- Operations --------------------------------------------------------------------------------------


    def __mul__(self,W):
        """
        Multiply this SO3vec with a list of weight vectors from the right.
        """
        R=SO3vec()
        for l in range(0,maxl):
            R.parts.append(self.parts[l]*W.parts[l])
        return R


    def FullCGproduct(self,y,maxl=-1,dummy=0):
        """
        Compute the full Clesbsch--Gordan product of this SO3vec with another SO3vec y.
        """
        r=SO3vec()
        r.parts=list(SO3vec_FullCGproductFn.apply(len(self.parts),len(y.parts),maxl,*(self.parts+y.parts)))
        return r


    def CGproduct(self,y,maxl):
        """
        Compute the full Clesbsch--Gordan product of this SO3Fvec with another SO3Fvec y.
        """
        tau=CGproductType(self.tau(),y.tau(),maxl)
        r=SO3vec.zeros(tau)
        offs=[0]*len(tau)
        for l1 in range(0,len(self.parts)):
            for l2 in range(0,len(y.parts)):
                for l in range(abs(l1-l2),min(l1+l2,maxl)+1):
                    r.parts[l].addCGproduct(self.parts[l1],y.parts[l2],offs[l]) # need different fn
                    offs[l]+=self.parts[l1].getN()*y.parts[l2].getN()        
        return r


    def DiagCGproduct(self,y,maxl,dummy=0):
        """
        Compute the diagonal Clesbsch--Gordan product of this SO3vec with another SO3vec y.
        """
        r=SO3vec()
        r.parts=list(SO3vec_DiagCGproductFn.apply(len(self.parts),len(y.parts),maxl,*(self.parts+y.parts)))
        return r


    def BlockwiseCGproduct(self,y,maxl,dummy=0):
        """
        Compute the blockwise Clesbsch--Gordan product of this SO3vec with another SO3vec y.
        """
        r=SO3vec()
        r.parts=list(SO3vec_BlockwiseCGproductFn.apply(len(self.parts),len(y.parts),maxl,*(self.parts+y.parts)))
        return r


    ## ---- I/O ----------------------------------------------------------------------------------------------

        
    def __str__(self):
        r=""
        for p in self.parts:
            r+=p.__str__()+"\n"
        return r




        


## ---- Other functions --------------------------------------------------------------------------------------

    
def tau_type(x):
    r=[]
    for t in x:
        r.append(t.getn())
    return r

def tau_type_blocked(x):
    r=[]
    for t in x:
        r.append(t.get_bsize())
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

def DiagCGproductType(x,maxl=-1):
    if maxl==-1:
        maxl=2*len(x)-2
    r=[0]*(maxl+1)
    for l1 in range(0,len(x)):
        for l2 in range(0,len(x)):
            for l in range(abs(l1-l2),min(l1+l2,maxl)+1):
                r[l]+=1
    return r


def MakeZeroSO3parts(tau):
    R=[]
    for l in range(0,len(tau)):
        R.append(SO3part.zeros(l,tau[l]))
    return R

def MakeZeroBlockedSO3parts(tau,nb):
    R=[]
    for l in range(0,len(tau)):
        R.append(SO3part.zeros(l,tau[l]*nb))
        R[len(R)-1].bsize=tau[l]
    return R



def FullCGproduct(x,y,a=-1,b=0):
    return x.FullCGproduct(y,a,b)

def DiagCGproduct(x,y,a=-1,b=0):
    return x.DiagCGproduct(y,a,b)

def BlockwiseCGproduct(x,y,a=-1,b=0):
    return x.BlockwiseCGproduct(y,a,b)



## ---- Autograd functions -----------------------------------------------------------------------------------


class SO3part_FullCGproductFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y,l,offs=0):
        ctx.save_for_backward(x,y)
        r=SO3part.zeros(l,x.getn()*y.getn())
        _SO3part.view(r).addFullCGproduct(_SO3part.view(x),_SO3part.view(y),offs)
        return r

    @staticmethod
    def backward(ctx, grad):
        x,y=ctx.saved_tensors
        grad_x=grad_y=None
        #print("backward")
        if ctx.needs_input_grad[0]:
            grad_x=torch.zeros_like(x)
            #print("backward to x")
            print(grad_x)
            _SO3part.view(grad_x).addFullCGproduct_back0(_SO3part.view(grad),_SO3part.view(y),offs)
        if ctx.needs_input_grad[1]:
            grad_y=torch.zeros_like(y)
            #print("backward to y")
            _SO3part.view(grad_y).addFullCGproduct_back1(_SO3part.view(grad),_SO3part.view(x),offs)
        return grad_x, grad_y, None, None


class SO3part_BlockwiseCGproductFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y,l,offs=0):
        ctx.save_for_backward(x,y)
        nb=x.get_nblocks()
        assert(y.get_nblocks()==nb)
        r=SO3part.zeros(l,nb*x.get_bsize()*y.get_bsize())
        _SO3part.view(r).addBlockwiseCGproduct(_SO3part.view(x),_SO3part.view(y),nb,offs)
        return r

    @staticmethod
    def backward(ctx, grad):
        x,y=ctx.saved_tensors
        grad_x=grad_y=None
        if ctx.needs_input_grad[0]:
            grad_x=torch.zeros_like(x)
            _SO3part.view(grad_x).addBlockwiseCGproduct_back0(_SO3part.view(grad),_SO3part.view(y),nb,offs)
        if ctx.needs_input_grad[1]:
            grad_y=torch.zeros_like(y)
            _SO3part.view(grad_y).addBlockwiseCGproduct_back1(_SO3part.view(grad),_SO3part.view(x),nb,offs)
        return grad_x, grad_y, None, None


class SO3part_DiagCGproductFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y,l,offs=0):
        ctx.save_for_backward(x,y)
        assert(x.getn()==y.getn())
        r=SO3part.zeros(l,x.getn())
        _SO3part.view(r).addBlockwiseCGproduct(_SO3part.view(x),_SO3part.view(y),offs)
        return r

    @staticmethod
    def backward(ctx, grad):
        x,y=ctx.saved_tensors
        grad_x=grad_y=None
        if ctx.needs_input_grad[0]:
            assert(grad.getn()==y.getn())
            grad_x=torch.zeros_like(x)
            _SO3part.view(grad_x).addDiagCGproduct_back0(_SO3part.view(grad),_SO3part.view(y),offs)
        if ctx.needs_input_grad[1]:
            assert(grad.getn()==y.getn())
            grad_y=torch.zeros_like(y)
            _SO3part.view(grad_y).addDiagCGproduct_back1(_SO3part.view(grad),_SO3part.view(x),offs)
        return grad_x, grad_y, None, None



class SO3vec_FullCGproductFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,k1,k2,maxl,*args):
        ctx.k1=k1
        ctx.k2=k2
        ctx.maxl=maxl
        ctx.save_for_backward(*args)

        tau=CGproductType(tau_type(args[0:k1]),tau_type(args[k1:k1+k2]),maxl)
        r=MakeZeroSO3parts(tau)

        _x=_SO3vec.view(args[0:k1]);
        _y=_SO3vec.view(args[k1:k1+k2]);
        _r=_SO3vec.view(r)
        _r.addCGproduct(_x,_y,maxl)

        return tuple(r)

    @staticmethod
    def backward(ctx,*args):
        print("backward")

        k1=ctx.k1
        k2=ctx.k2
        maxl=ctx.maxl

        inputs=ctx.saved_tensors
        assert len(inputs)==k1+k2, "Wrong number of saved tensors."

        grads=[None,None,None]
        for i in range(k1+k2):
            grads.append(torch.zeros_like(inputs[i]))

        _x=_SO3vec.view(inputs[0:k1]);
        _y=_SO3vec.view(inputs[k1:k1+k2]);

        _g=_SO3vec.view(args);
        _xg=_SO3vec.view(grads[3:k1+3]);
        _yg=_SO3vec.view(grads[k1+3:k1+k2+3]);

        _xg.addCGproduct_back0(_g,_y,maxl)
        _yg.addCGproduct_back1(_g,_x,maxl)

        return tuple(grads)



class SO3vec_BlockwiseCGproductFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,k1,k2,maxl,*args):
        ctx.k1=k1
        ctx.k2=k2
        ctx.maxl=maxl
        ctx.save_for_backward(*args)

        nb=args[0].get_nblocks()
        tau=CGproductType(tau_type_blocked(args[0:k1]),tau_type_blocked(args[k1:k1+k2]),maxl)
        r=MakeZeroBlockedSO3parts(tau,nb)

        _x=_SO3vec.view(args[0:k1]);
        _y=_SO3vec.view(args[k1:k1+k2]);
        _r=_SO3vec.view(r)
        _r.addBlockwiseCGproduct(_x,_y,nb,maxl)

        return tuple(r)

    @staticmethod
    def backward(ctx,*args):
        print("backward")

        k1=ctx.k1
        k2=ctx.k2
        maxl=ctx.maxl

        inputs=ctx.saved_tensors
        assert len(inputs)==k1+k2, "Wrong number of saved tensors."

        grads=[None,None,None]
        for i in range(k1+k2):
            grads.append(torch.zeros_like(inputs[i]))

        _x=_SO3vec.view(inputs[0:k1]);
        _y=_SO3vec.view(inputs[k1:k1+k2]);

        _g=_SO3vec.view(args);
        _xg=_SO3vec.view(grads[3:k1+3]);
        _yg=_SO3vec.view(grads[k1+3:k1+k2+3]);

        _xg.addBlockwiseCGproduct_back0(_g,_y,nb,maxl)
        _yg.addBlockwiseCGproduct_back1(_g,_x,nb,maxl)

        return tuple(grads)



class SO3vec_DiagCGproductFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,k1,k2,maxl,*args):
        ctx.k1=k1
        ctx.k2=k2
        assert(k1==k2)
        ctx.maxl=maxl
        ctx.save_for_backward(*args)

        _tau=tau_type(args[0:k1])
        assert _tau==tau_type(args[k1:k1+k2])
        tau=DiagCGproductType(_tau,maxl)
        r=MakeZeroSO3parts(tau)

        _x=_SO3vec.view(args[0:k1]);
        _y=_SO3vec.view(args[k1:k1+k2]);
        _r=_SO3vec.view(r)
        _r.addDiagCGproduct(_x,_y,maxl)

        return tuple(r)

    @staticmethod
    def backward(ctx,*args):

        k1=ctx.k1
        k2=ctx.k2
        maxl=ctx.maxl

        inputs=ctx.saved_tensors
        assert len(inputs)==k1+k2, "Wrong number of saved tensors."

        grads=[None,None,None]
        for i in range(k1+k2):
            grads.append(torch.zeros_like(inputs[i]))

        _x=_SO3vec.view(inputs[0:k1]);
        _y=_SO3vec.view(inputs[k1:k1+k2]);

        _g=_SO3vec.view(args);
        _xg=_SO3vec.view(grads[3:k1+3]);
        _yg=_SO3vec.view(grads[k1+3:k1+k2+3]);

        _xg.addDiagCGproduct_back0(_g,_y,maxl)
        _yg.addDiagCGproduct_back1(_g,_x,maxl)

        return tuple(grads)




        #r.requires_grad_()
        #x.retain_grad()
    # inplace operations won't work
    #def addFullCGproduct(self,x,y,offs=0):
    #    "Add CGproduct(x,y) with offset offs"
    #    if (isinstance(x,SO3part) and isinstance(y,SO3part)):
    #        SO3part_addFullCGproductFn.apply(self,x,y,offs)
    #    else:
    #        raise TypeError('Type of each argument in addFullCGproduct(SO3part,SO3part) must be SO3part.')


        #print("forward")
        #r.requires_grad_()
        #x.retain_grad()
