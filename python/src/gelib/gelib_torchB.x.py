import torch

from gelib_base import SO3part as _SO3part



class ctens(torch.Tensor):

    def __init__(self,_T):
        self=_T

    @staticmethod
    def zeros(_dims):
        return ctens(torch.zeros([2]+_dims))
    
    @staticmethod
    def randn(_dims):
        return ctens(torch.randn([2]+_dims))

#    def dim(self):
#        return super().dim()-1

#    def size(self,_dim=-1):
#        if _dim==-1:
#            return super().size()[1:]
#        else:
#            return super().size(_dim+1)
    
    def mprod(self,y):
        assert self.dim()==3, "Must have two tensor dimensions"
        assert y.dim()==3, "Must have two tensor dimensions"
        assert self.size(2)==y.size(1), "Inner dimension mismatch"
        r=ctens.zeros([self.size(1),y.size(2)])
        rr=r.select(0,0)
        rr=torch.matmul(self.select(0,0),y.select(0,0))-torch.matmul(self.select(0,1),y.select(0,1))
        ri=r.select(0,0)
        ri=torch.matmul(self.select(0,0),y.select(0,1))+torch.matmul(self.select(0,1),y.select(0,0))
        return r;

 
    def mix(self,i,M):
        "Multiply along the i'th tensor dimension by the matrix W"
        assert M.dim()==3,"M must be a ctens matrix"
        assert i==dim()-2,"Currently mixing is only supported along last dimension"
        assert self.size(i+1)==M.size(1), "Inner dimension mismatch"
        ix=self.size()
        ix[len(ix)-1]=M.size(2)
        mi=1
        for j in range(1,i+1):
            mi*=self.size(j)
        mj=self.size(j+1)
        mk=M.size(2)
        
        r=ctens.zeros(ix)
        rr=r.select(0,0).reshape(mi,mk)
        ri=r.select(0,0).reshape(mi,mk)
        xr=self.select(0,0).reshape(mi,mj)
        xi=self.select(0,1).reshape(mi,mj)
        rr=torch.matmul(xr,M.select(0,0))-torch.matmul(xi,M.select(0,1))
        ri=torch.matmul(xr,M.select(0,1))+torch.matmul(xi,M.select(0,0))
        return r;


    def mprod2(M):
        """
        Contract the -2 dimension of this tensor with the -1 dimension of M
        """
        assert self.dim()==4, "Must be third order"
        assert M.dim()==4, "Must be third order"

        I=self.size(1)
        J=self.size(2)
        K=M.size(2)
        A=self.size(-1)
        assert M.size(-1)==A, "Last dimension of tensors must match"
        assert J==M.size(1), "Inner dimsion mismatch"

        r=ctens.zeros([I,K,A])
        for a in range(A):
            rr=ctens.select(0,0).select(-1,a)
            ri=ctens.select(0,1).select(-1,a)
            rr=torch.matmul(self.select(0,0).select(-1,a),y.select(0,0).select(-1,a))-torch.matmul(self.select(0,1).select(-1,a),y.select(0,1).select(-1,a))
            ri=torch.matmul(self.select(0,0).select(-1,a),y.select(0,1).select(-1,a))+torch.matmul(self.select(0,1).select(-1,a),y.select(0,0).select(-1,a))
        return r;
            
        
    def __mul__(self,y):
        return self.mprod(y)

    def __str__(self):
        return str(torch.Tensor(self))
        #return super().__str__()

    def __repr__(self):
        return "<ctens("+str(self.size())+")>"


## ---- SO3part --------------------------------------------------------------------------------------------


class SO3part(ctens):
    """
    A collection of vectors that transform according to a specific irreducible representation of SO(3).
    """
    
    def __init__(self,_T):
        self=_T

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

    def getl(self):
        "Return l, the index of the representation."
        return int((self.size(1)-1)/2)

    def getn(self):
        "Return n, the number of vectors."
        return self.size(2)

    def __mul__(self,y):
        return SO3part(self.mprod(y))

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

    @staticmethod
    def zeros(_tau):
        """
        Construct a zero SO3vec object of given type _tau.

        Parameters
        ----------

        _tau: :class:`list`
           A list of integers determining how many components of index l=0,1,2,... the SO3vec has.

        """
        R=SO3vec()
        for l in range(0,len(_tau)):
            R.parts.append(SO3part.zeros(l,_tau[l]))
        return R

    @staticmethod
    def randn(_tau):
        """
        Construct a random SO3vec object of given type _tau.

        Parameters
        ----------

        _tau: :class:`list`
           A list of integers determining how many components of index l=0,1,2,... the SO3vec has.

        """
        R=SO3vec()
        for l in range(0,len(_tau)):
            R.parts.append(SO3part.randn(l,_tau[l]))
        return R

    @staticmethod
    def Frandn(maxl):
        R=SO3vec()
        for l in range(0,maxl+1):
            R.parts.append(SO3part.randn(l,2*l+1))
        return R


    def tau(self):
        """
        Return the 'type' of the SO3vec, i.e., how many components it has corresponding to l=0,1,2,...
        """
        r=[]
        for l in range(0,len(self.parts)):
            r.append(self.parts[l].getn())
        return r


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
        
    def __str__(self):
        r=""
        for p in self.parts:
            r+=p.__str__()+"\n"
        return r



## ---- SO3Fpart -------------------------------------------------------------------------------------------


class SO3Fpart(ctens):
    """
    A tensor of Fourier SO3parts, i.e., an SO3part where the number of vectors is a multiple of 2*l+1. 
    """
    
    def __init__(self,_T):
        self=_T

    @staticmethod
    def zeros(l,n):
        """
        Create an Fourier SO(3)-part consisting of n Fourier matrices of size (2*l+1)*(2*l+1). 
        The matrices are initialized to zero, resulting in (2+l+1)*(2*l+1)*n dimensional complex tensor of
        zeros stored as a ctens object.
        """        
        return SO3part(ctens.zeros([2*l+1,2*l+1,n]))

    @staticmethod
    def randn(l,n):
        """
        Create an Fourier SO(3)-part consisting of n Fourier matrices of size (2*l+1)*(2*l+1). 
        The matrices are initialized to random gaussians, resulting in (2+l+1)*(2*l+1)*n dimensional
        complex tensor of zeros stored as a ctens object.
        """        
        return SO3part(ctens.randn([2*l+1,2*l+1,n]))

    def getl(self):
        "Return l, the index of the representation."
        return int((self.size(1)-1)/2)

    def getn(self):
        "Return n, the third dimension of the tensor."
        return self.size(3)

    def getN(self):
        return self.size(2)*self.size(3)
    
    def mix(self,W):
        "Multiply this SO3Fpart with a weight matrix along the 3rd dimension"
        return SO3Fpart(self.mix(2,W))

    def convolve(self,y):
        "Convolve this SO3Fpart with another SO3Fpart"
        return SO3Fpart(self.mprod2(y))

    def addCGproduct(self,x,y,offs=0):
        "Add CGproduct(x,y) with offset offs"
        if isinstance(x,SO3Fpart) && isinstance(y,SO3Fpart):
            SO3Fpart_addCGproductFn.apply(self,x,y,offs)
        else:
            raise TypeError('Type of each argument in addCGproduct(SO3part,SO3part) must be SO3Fpart.')
                
    def __str__(self):
        return super().__str__()

    def __repr__(self):
        return "<SO3Fpart("+str(self.getl())+","+str(self.getn())+")>"



class SO3Fpart_addCGproductFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,r,x,y,offs):
        ctx.save_for_backward(x,y,l);
        return SO3Fpart(GElib.SO3part_addCGproduct(_SO3part.view(r),_SO3part.view(x),_SO3part.view(y),offs).torch())

    @staticmethod
    def backward(ctx, grad):
        x,y,l=ctx.saved_tensors
        grad_x=grad_y=None
        if ctx.needs_input_grad[0]:
            grad_x=torch.zeros_like(x)
            _SO3part.view(grad_x).SO3part_addCGproduct0(_SO3part.view(grad),_SO3part.view(y))
        if ctx.needs_input_grad[1]:
            grad_y=torch.zeros_like(y)
            _SO3part.view(grad_y).SO3part_addCGproduct1(_SO3part.view(grad),_SO3part.view(x))



## ---- SO3Fvec ---------------------------------------------------------------------------------------------
    

class SO3Fvec:
    """
    An SO(3)-covariant vector consisting of a sequence of SO3Fpart objects. This means that the SO3Fvec has
    an interpretation as the Fourier transform of a vector valued function on SO(3).     
    """
    
    def __init__(self):
        self.parts=[]

    @staticmethod
    def zeros(maxl,n):
        """
        Construct a zero SO3Fvec to store the Fourier transform of n functions up to l=maxl.
        """
        R=SO3vec()
        for l in range(0,len(maxl+1)):
            R.parts.append(SO3Fpart.zeros(l,n))
        return R


    @staticmethod
    def randn(maxl,n):
        """
        Construct a zero SO3Fvec to store the Fourier transform of n functions up to l=maxl.
        """
        R=SO3vec()
        for l in range(0,len(maxl+1)):
            R.parts.append(SO3Fpart.randn(l,n))
        return R


    @staticmethod
    def zeros(_nu):
        """
        Construct a zero SO3Fvec of type nu.

        Parameters
        ----------

        _nu: :class:`list`
           A list of integers, where nu[l] determines the size of the third dimension of the l'th part,
           so the l'th part has (2*l+1)*nu[l] vectors. 

        """
        R=SO3vec()
        for l in range(0,len(_nu)):
            R.parts.append(SO3Fpart.zeros(l,_nu[l]))
        return R


    @staticmethod
    def randn(_nu):
        """
        Construct a zero SO3Fvec object of given type _nu initialized to random gaussians.

        Parameters
        ----------

        _nu: :class:`list`
           A list of integers, where nu[l] determines the size of the third dimension of the l'th part,
           so the l'th part has (2*l+1)*nu[l] vectors. 

        """
        R=SO3vec()
        for l in range(0,len(_nu)):
            R.parts.append(SO3Fpart.randn(l,_nu[l]))
        return R


    def tau(self):
        """
        Return the 'type' of the SO3Fvec, i.e., how many components it has corresponding to l=0,1,2,...
        """
        r=[]
        for l in range(0,len(self.parts)):
            r.append(self.parts[l].getN())
        return r


    def mix(self,W):
        """
        Multiply this SO3Fvec with a list of weight vectors from the right along the 3rd dimension.
        """
        R=SO3Fvec()
        for l in range(0,maxl):
            R.parts.append(self.parts[l]*W.parts[l])
        return R


    def convolve(self,W):
        """
        Multiply each SO3Fpart with the corresponding SO3Fpart along the 2nd dimension
        """
        R=SO3Fvec()
        for l in range(0,maxl):
            R.parts.append(self.parts[l].convolve(W.parts[l]))
        return R


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

        


## ---- SO3weights -------------------------------------------------------------------------------------------


class ctens_list:
    """
    Class to store a list of weight matrices that can be multiply an :class:`SO3vec` object
    """

    def __init__(self):
        self.parts=[]

    @staticmethod
    def zeros(_dims):
        """
        Construct a Weights objects consisting of zero matrices. Each matrix is a complex matrix of
        type :class:`ctens`.

        Parameters
        ----------

        _dims: List of [i,j] pairs. If the l'th pair is [i,j], the l'th matrix will be an i times j
        dimensional complex matrix. 
        """
        r=ctens_list()
        for p in _dims:
            r.parts.append(ctens.zeros([p[0],p[1]]))
        return r

    def __str__(self):
        r=""
        for l in range(len(self.parts)):
            r+="Part "+str(l)+":\n"+str(self.parts[l])+"\n"
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
