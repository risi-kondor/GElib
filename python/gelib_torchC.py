import torch

from gelib_base import SO3partB as _SO3partB
from gelib_base import SO3Fvec as _SO3Fvec




## ----------------------------------------------------------------------------------------------------------
## ---- SO3Fvec ----------------------------------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------------------
    

class SO3Fvec:
    """
    An SO(3)-covariant vector that represents the Fourier transform of b different functions on SO(3)
    """
    
    def __init__(self):
        self.parts=[]


    ## ---- Static constructors ------------------------------------------------------------------------------


    @staticmethod
    def zeros(b,maxl,device=0):
        "Construct a zero SO3Fvec object with l ranging from 0 to maxl."
        R=SO3Fvec()
        for l in range(0,maxl+1):
            if device==0:
                R.parts.append(torch.zeros(b,2*l+1,2*l+1,2,dtype=torch.float))
            else:
                R.parts.append(torch.zeros(b,2*l+1,2*l+1,2,dtype=torch.float,device=cuda0))
        return R

    @staticmethod
    def randn(b,maxl,device=0):
        "Construct a zero SO3Fvec object  with l ranging from 0 to maxl."
        R=SO3Fvec()
        for l in range(0,maxl+1):
            if device==0:
                R.parts.append(torch.randn(b,2*l+1,2*l+1,2,dtype=torch.float))
            else:
                R.parts.append(torch.randn(b,2*l+1,2*l+1,2,dtype=torch.float,device=cuda0))             
        return R

    @staticmethod
    def Fzeros_like(x):
        R=SO3Fvec()
        b=x.parts[0].dim(0)
        for l in range(0,len(x.parts)):
            R.parts.append(torch.zeros(b,2*l+1,2*l+1,2,dtype=torch.float))
        return R;
                           
                       
    
    ## ---- Access -------------------------------------------------------------------------------------------


    ## ---- Operations ---------------------------------------------------------------------------------------

        
    ## ---- Products -----------------------------------------------------------------------------------------


    def Fproduct(self,y,maxl=-1):
        """
        Compute the Fourier space product of this SO3Fvec with another SO3Fvec y.
        """
        r=SO3Fvec()
        r.parts=list(SO3Fvec_FproductFn.apply(len(self.parts),len(y.parts),maxl,*(self.parts+y.parts)))
        return r


    ## ---- I/O ----------------------------------------------------------------------------------------------

        
    def __str__(self):
        u=_SO3Fvec.view(self.parts)
        return u.__str__()
        #r=""
        #for p in self.parts:
        #    r+=p.__str__()+"\n"
        #return r


## ----------------------------------------------------------------------------------------------------------
## ---- Other functions --------------------------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------------------

    
def Fproduct(x,y,a=-1):
    return x.Fproduct(y,a)


def makeZeroSO3Fparts(b,maxl):
    R=[]
    for l in range(0,maxl+1):
        R.append(torch.zeros(b,2*l+1,2*l+1,2,dtype=torch.float))
    return R



## ----------------------------------------------------------------------------------------------------------
## ---- Autograd functions -----------------------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------------------


class SO3Fvec_FproductFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,k1,k2,_maxl,*args):
        ctx.k1=k1
        ctx.k2=k2
        ctx.save_for_backward(*args)

        b=args[0].size(0)
        if _maxl==-1:
            maxl=k1+k2-2
        else:
            maxl=_maxl

        r=makeZeroSO3Fparts(b,maxl)

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


