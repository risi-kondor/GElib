import torch

from gelib_base import SO3partB as _SO3partB
from gelib_base import SO3vecB as _SO3vecB
from gelib_base import SO3Fvec as _SO3Fvec
#from gelib_base import SO3partD as _SO3partD
#from gelib_base import SO3vecD as _SO3vecD


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
        return SO3partArr(_SO3partB.view(self).apply(R).torch())


    def gather(self,_mask):
        """
        Gather the elements of this SO3partArr into a new SO3partArr according to the mask
        """
        return SO3partArr(SO3partArr_GatherFn.apply(_mask,self))


    ## ---- I/O ----------------------------------------------------------------------------------------------

        
    def __str__(self):
        u=_SO3partB.view(self)
        return u.__str__()
