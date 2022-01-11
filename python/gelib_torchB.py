import torch

from gelib_base import SO3part as _SO3part
from ctens import *


## ---- SO3part --------------------------------------------------------------------------------------------


class SO3part(ctens):
    """
    A collection of vectors that transform according to a specific irreducible representation of SO(3).
    The parts are stacked into a complex matrix of type ctens. 
    """
    
    def __init__(self,_T):
        self=_T


    ## ---- Static constructors -----------------------------------------------------------------------------

        
    @staticmethod
    def zeros(l,n):
        """
        Create an SO(3)-part consisting of n vectors transforming according to the l'th irrep of SO(3).
        The vectors are initialized to zero, resulting in (2+l+1)*n dimensional complex matrix of zeros
        stored as a ctens object.
        """        
        return SO3part(ctens.zeros([2*l+1,n]))

    @staticmethod
    def randn(l,n):
        """
        Create an SO(3)-part consisting of n vectors transforming according to the l'th irrep of SO(3).
        The vectors are initialized as random gaussian vectors, resulting in (2+l+1)*n dimensional
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
        return SO3part(ctens.zeros([2*l+1,(2*l+1)*n],2*l+1))

    @staticmethod
    def Frandn(l,n):
        """
        Create an SO(3)-part consisting of n blocks of (2l+1) vectors each transforming according to the
        l'th irrep of SO(3). The vectors are initialized as random Gaussian vectors, giving a
        (2+l+1)*(2l+1)*n dimensional random complex matrix of stored as a ctens object.
        """        
        return SO3part(ctens.randn([2*l+1,(2*l+1)*n],2*l+1))


    ## ---- Access ------------------------------------------------------------------------------------------


    def getl(self):
        "Return l, the index of the representation."
        return int((self.size(1)-1)/2)

    def getn(self):
        "Return n, the number of vectors."
        return self.size(2)


    ## ---- Operations ---------------------------------------------------------------------------------------


    def __mul__(self,y):
        return SO3part(self.matmul(y))

    def mix(self,W):
        "Multiply this SO3Fpart with a weight matrix along the 3rd dimension"
        return SO3part(self.mix_blocks(2,W))

    def convolve(self,y):
        "Convolve this SO3Fpart with another SO3Fpart"
        return SO3Fpart(self.matmul_each_block_by_corresponding_block(y))

    def addFullCGproduct(self,x,y,offs=0):
        "Add CGproduct(x,y) with offset offs"
        if (isinstance(x,SO3part) and isinstance(y,SO3part)):
            SO3part_addFullCGproductFn.apply(self,x,y,offs)
        else:
            raise TypeError('Type of each argument in addFullCGproduct(SO3part,SO3part) must be SO3part.')


    ## ---- I/O ----------------------------------------------------------------------------------------------


    def __str__(self):
        return super().__str__()

    def __repr__(self):
        return "<SO3part("+str(self.getl())+","+str(self.getn())+")>"



## ---- SO3vec ----------------------------------------------------------------------------------------------
    

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
        "Construct a zero SO3vec object of maxl parts corresponding to the union of _n Fourier transforms."
        R=SO3vec()
        for l in range(0,_maxl):
            R.parts.append(SO3part.Fzeros(_n))
        return R

    @staticmethod
    def Frandn(_maxl,_n):
        "Construct a random SO3vec object corresponding to the union of _n Fourier transforms."
        R=SO3vec()
        for l in range(0,_maxl):
            R.parts.append(SO3part.Frandn(_n))
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


    def CGproduct(self,y):
        """
        Compute the Clesbsch--Gordan product of this SO3vec with another SO3vec y.
        """
        tau=CGproductType(self.tau(),y.tau())
        r=SO3vec.zeros(tau)
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


    def DiagCGproduct(self,y,maxl):
        """
        Compute the diagonal Clesbsch--Gordan product of this SO3Fvec with another SO3Fvec y.
        """
        r=SO3vec.zeros(tau)
        for l1 in range(0,len(self.parts)):
            r.parts[l].addDiagCGproduct(self.parts[l1],y.parts[l2]) 
        return r

    ## ---- I/O ----------------------------------------------------------------------------------------------

        
    def __str__(self):
        r=""
        for p in self.parts:
            r+=p.__str__()+"\n"
        return r




        


## ---- Other functions --------------------------------------------------------------------------------------
    

def CGproductType(x,y,maxl=-1):
    if maxl==-1:
        maxl=len(x)+len(y)-2
    r=[0]*(maxl+1)
    for l1 in range(0,len(x)):
        for l2 in range(0,len(y)):
            for l in range(abs(l1-l2),min(l1+l2,maxl)+1):
                r[l]+=x[l1]*y[l2]
    return r


## ---- Autograd functions -----------------------------------------------------------------------------------


class SO3part_addFullCGproductFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,r,x,y,offs):
        ctx.mark_dirty(r)
        ctx.save_for_backward(x,y)
        print("forward")
        r.requires_grad_()
        return r
        #return SO3part(GElib.SO3partCGproduct(_SO3part.view(x),_SO3part.view(y)).torch())

    @staticmethod
    def backward(ctx, grad):
        x,y=ctx.saved_tensors
        grad_x=grad_y=None
        print("back")
        if ctx.needs_input_grad[0]:
            grad_x=torch.zeros_like(x)
            print("backward to x")
            #grad_x.cview().SO3partCGproduct0(grad.cview(),y.cview())
        if ctx.needs_input_grad[1]:
            grad_y=torch.zeros_like(y)
            print("backward to y")
            #grad_y.cview().SO3partCGproduct1(grad.cview(),x.cview())
        return grad_x, grad_x, grad_y, None
