import torch

from gelib_base import SO3partB as _SO3partB
from gelib_base import SO3vecB as _SO3vecB
from gelib_base import SO3Fvec as _SO3Fvec
from gelib_base import SO3partD as _SO3partD
from gelib_base import SO3vecD as _SO3vecD


## ----------------------------------------------------------------------------------------------------------
## ---- SO3part ---------------------------------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------------------

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
    def zeros(N,b,l,n,_dev=0):
        """
        Create an SO(3)-part consisting of N*b lots of n vectors transforming according to the l'th irrep of SO(3).
        The vectors are initialized to zero, resulting in an b*(2+l+1)*n dimensional complex tensor of zeros.
        """        
        if _dev==0:
            return SO3partArr(torch.zeros([N,b,2*l+1,n,2]))
        else:
            return SO3partArr(torch.zeros([N,b,2*l+1,n,2])).cuda()


    @staticmethod
    def randn(N,b,l,n,_dev=0):
        """
        Create an SO(3)-part consisting of N*b lots of n vectors transforming according to the l'th irrep of SO(3).
        The vectors are initialized as random gaussian vectors, resulting in an b*(2+l+1)*n dimensional random
        complex tensor.
        """
        if _dev==0:        
            return SO3partArr(torch.randn([N,b,2*l+1,n,2]))
        else:
            return SO3partArr(torch.randn([N,b,2*l+1,n,2],device='cuda'))


    @staticmethod
    def Fzeros(N,b,l,_dev=0):
        """
        Create an SO(3)-part corresponding to the l'th matrix in the Fourier transform of a function on SO(3).
        This gives a N*b*(2+l+1)*(2l+1) dimensional complex tensor. 
        """
        if _dev==0:        
            return SO3partArr(torch.zeros([N,b,2*l+1,2*l+1,2]))
        else:
            return SO3partArr(torch.zeros([N,b,2*l+1,2*l+1,2])).cuda() # why doesn't device='cuda' work?

    @staticmethod
    def Frandn(N,b,l,_dev=0):
        """
        Create an SO(3)-part corresponding to the l'th matrix in the Fourier transform of a function on SO(3).
        This gives a b*(2+l+1)*(2l+1) dimensional complex random tensor. 
        """
        if _dev==0:        
            return SO3partArr(torch.randn([N,b,2*l+1,2*l+1,2]))
        else:
            return SO3partArr(torch.randn([N,b,2*l+1,2*l+1,2],device='cuda'))


    ## ---- Access ------------------------------------------------------------------------------------------


    ## ---- Operations --------------------------------------------------------------------------------------


    def rotate(self,R):
        return SO3partArr(_SO3partD.view(self).apply(R).torch())


    def gather(self,_mask):
        """
        Gather the elements of this SO3partArr into a new SO3partArr according to the mask
        """
        return SO3partArr(SO3partArr_GatherFn.apply(_mask,self))


    ## ---- I/O ----------------------------------------------------------------------------------------------

        
    def __str__(self):
        u=_SO3partD.view(self)
        return u.__str__()




## ----------------------------------------------------------------------------------------------------------
## ---- SO3vec ----------------------------------------------------------------------------------------------
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
    def zeros(N,b,_tau,_dev=0):
        "Construct a zero SO3vec object of given type _tau."
        R=SO3vecArr()
        for l in range(0,len(_tau)):
            R.parts.append(SO3partArr.zeros(N,b,l,_tau[l],_dev))
        return R

    @staticmethod
    def randn(N,b,_tau,_dev=0):
        "Construct a random SO3vec array of given type _tau."
        R=SO3vecArr()
        for l in range(0,len(_tau)):
            R.parts.append(SO3partArr.randn(N,b,l,_tau[l],_dev))
        return R

    @staticmethod
    def Fzeros(N,b,maxl,_dev=0):
        "Construct an SO3vec array corresponding the to the Forier matrices 0,1,...maxl of b functions on SO(3)."
        R=SO3vecArr()
        for l in range(0,maxl+1):
            R.parts.append(SO3partArr.Fzeros(N,b,l,_dev))
        return R

    @staticmethod
    def Frandn(N,b,maxl,_dev=0):
        "Construct a zero SO3Fvec array with l ranging from 0 to maxl."
        R=SO3vecArr()
        for l in range(0,maxl+1):
            R.parts.append(SO3partArr.Frandn(N,b,l,_dev))            
        return R

    @staticmethod
    def zeros_like(x):
        R=SO3vecrArr()
        b=x.parts[0].dim(0)
        for l in range(0,len(x.parts)):
            R.parts.append(SO3partArr(torch.zeros_like(x.parts[l])))
        return R;
                           
                       
    
    ## ---- Access -------------------------------------------------------------------------------------------


    def tau(self):
        "Return the 'type' of the SO3vec, i.e., how many components it has corresponding to l=0,1,2,..."
        r=[]
        for l in range(0,len(self.parts)):
            r.append(self.parts[l].size(3))
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

        
    def __str__(self):
        u=_SO3vecD.view(self.parts)
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
        r.append(t.size(3))
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


def MakeZeroSO3partArrs(N,b,tau,_dev=0):
    R=[]
    for l in range(0,len(tau)):
        R.append(SO3partArr.zeros(N,b,l,tau[l],_dev))
    return R

def makeZeroSO3FpartArrs(N,b,maxl,_dev=0):
    R=[]
    for l in range(0,maxl+1):
        R.append(SO3part.Fzeros(N,b,l,_dev))
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

        N=args[0].size(0)
        b=args[0].size(1)
        tau=CGproductType(tau_type(args[0:k1]),tau_type(args[k1:k1+k2]),maxl)
        dev=int(args[0].is_cuda)
        r=MakeZeroSO3partArrs(N,b,tau,dev)

        _x=_SO3vecD.view(args[0:k1]);
        _y=_SO3vecD.view(args[k1:k1+k2]);
        _r=_SO3vecD.view(r)
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

        _x=_SO3vecD.view(inputs[0:k1]);
        _y=_SO3vecD.view(inputs[k1:k1+k2]);

        _g=_SO3vecD.view(args);
        _xg=_SO3vecD.view(grads[3:k1+3]);
        _yg=_SO3vecD.view(grads[k1+3:k1+k2+3]);

        _xg.addCGproduct_back0(_g,_y)
        _yg.addCGproduct_back1(_g,_x)

        return tuple(grads)


class SO3vec_FproductFn(torch.autograd.Function): #todo

    @staticmethod
    def forward(ctx,k1,k2,_maxl,*args):
        ctx.k1=k1
        ctx.k2=k2
        ctx.save_for_backward(*args)

        N=args[0].size(0)
        b=args[0].size(1)
        if _maxl==-1:
            maxl=k1+k2-2
        else:
            maxl=_maxl
        dev=int(args[0].is_cuda)

        r=makeZeroSO3Fparts(N,b,maxl,dev)

        _x=_SO3Fvec.view(args[0:k1]);
        _y=_SO3Fvec.view(args[k1:k1+k2]);
        _r=_SO3Fvec.view(r)
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

        _x=_SO3Fvec.view(inputs[0:k1]);
        _y=_SO3Fvec.view(inputs[k1:k1+k2]);

        _g=_SO3Fvec.view(args);
        _xg=_SO3Fvec.view(grads[3:k1+3]);
        _yg=_SO3Fvec.view(grads[k1+3:k1+k2+3]);

        _xg.addFproduct_back0(_g,_y)
        _yg.addFproduct_back1(_g,_x)

        return tuple(grads)


class SO3vec_FmodsqFn(torch.autograd.Function): #todo

    @staticmethod
    def forward(ctx,k1,_maxl,*args):
        ctx.k1=k1
        #ctx.k2=k1
        ctx.save_for_backward(*args)

        b=args[0].size(0)
        if _maxl==-1:
            maxl=k1+k1-2
        else:
            maxl=_maxl
        dev=int(args[0].is_cuda)

        r=makeZeroSO3Fparts(b,maxl,dev)

        _x=_SO3Fvec.view(args[0:k1]);
        #_y=_SO3Fvec.view(args[k1:k1+k2]);
        _r=_SO3Fvec.view(r)
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

        _x=_SO3Fvec.view(inputs[0:k1]);
        #_y=_SO3Fvec.view(inputs[k1:k1+k2]);

        _g=_SO3Fvec.view(args);
        _xg=_SO3Fvec.view(grads[2:k1+2]);
        #_yg=_SO3Fvec.view(grads[k1+3:k1+k2+3]);

        _xg.addFproduct_back0(_g,_x)
        _xg.addFproduct_back1(_g,_x)

        return tuple(grads)



class SO3partArr_GatherFn(torch.autograd.Function): 

    @staticmethod
    def forward(ctx,_mask,x):

        ctx.mask=_mask
        N=x.size(0)
        b=x.size(1)
        l=int((x.size(2)-1)/2)
        n=x.size(3)
        dev=int(x.is_cuda)
        r=SO3partArr.zeros(N,b,l,n,dev)
        
        _x=_SO3partD.view(x)
        _r=_SO3partD.view(r)
        _r.gather(_x,_mask)

        return r

    @staticmethod
    def backward(ctx,yg):

        N=yg.size(0)
        b=yg.size(1)
        l=int((y.size(2)-1)/2)
        n=y.size(3)
        dev=int(y.is_cuda)
        r=MakeZeroSO3partArrs(N,b,l,n,dev)

        _x=_SO3partD.view(args)
        _r=_SO3partD.view(r)
        _r.gather(_x,ctx.mask.inv())

        return tuple([None]+r)


class SO3vecArr_GatherFn(torch.autograd.Function): 

    @staticmethod
    def forward(ctx,*args):

        ctx.mask=args[0]
        N=args[1].size(0)
        b=args[1].size(1)
        tau=tau_type(args[1:])
        dev=int(args[1].is_cuda)
        r=MakeZeroSO3partArrs(N,b,tau,dev)
        
        _x=_SO3vecD.view(args[1:])
        _r=_SO3vecD.view(r)
        _r.gather(_x,args[0])

        return tuple(r)

    @staticmethod
    def backward(ctx,*args):

        N=args[0].size(0)
        b=args[0].size(1)
        tau=tau_type(args)
        dev=int(args[0].is_cuda)
        r=MakeZeroSO3partArrs(N,b,tau,dev)

        _x=_SO3vecD.view(args)
        _r=_SO3vecD.view(r)
        _r.gather(_x,ctx.mask.inv())

        return tuple([None]+r)


