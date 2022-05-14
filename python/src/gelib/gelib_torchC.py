import torch

from gelib_base import SO3partB as _SO3partB
from gelib_base import SO3vecB as _SO3vecB
from gelib_base import SO3Fvec as _SO3Fvec


# ----------------------------------------------------------------------------------------------------------
# ---- SO3part ---------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------

class SO3part(torch.Tensor):
    """
    A collection of vectors that transform according to a specific irreducible representation of SO(3).
    The vectors are stacked into a third order tensor. The first index is the batch index, the second
    is m=-l,...,l, and the third index is the fragment index. 
    """

    def __init__(self, _T):
        self = _T

    # ---- Static constructors -----------------------------------------------------------------------------

    @staticmethod
    def zeros(b, l, n, _dev=0):
        """
        Create an SO(3)-part consisting of b lots of n vectors transforming according to the l'th irrep of SO(3).
        The vectors are initialized to zero, resulting in an b*(2+l+1)*n dimensional complex tensor of zeros.
        """
        if _dev == 0:
            return SO3part(torch.zeros([b, 2*l+1, n, 2]))
        else:
            return SO3part(torch.zeros([b, 2*l+1, n, 2])).cuda()

    @staticmethod
    def randn(b, l, n, _dev=0):
        """
        Create an SO(3)-part consisting of b lots of n vectors transforming according to the l'th irrep of SO(3).
        The vectors are initialized as random gaussian vectors, resulting in an b*(2+l+1)*n dimensional random
        complex tensor.
        """
        if _dev == 0:
            return SO3part(torch.randn([b, 2*l+1, n, 2]))
        else:
            return SO3part(torch.randn([b, 2*l+1, n, 2])).cuda()

    @staticmethod
    def spharm(l, x, y, z, _dev=0):
        """
        Return the spherical harmonics of the vector (x,y,z)
        """
        R = SO3part.zeros(1, l, 1)
        _SO3partB.view(R).add_spharm(x, y, z)
        if _dev > 0:
            return R.cuda()
        return R

    @staticmethod
    def spharmB(l, X, _dev=0):
        """
        Return the spherical harmonics of each row of the matrix X.
        """
        R = SO3part.zeros(X.size(0), l, 1)
        _SO3partB.view(R).add_spharmB(X)
        if _dev > 0:
            return R.cuda()
        return R

    @staticmethod
    def spharM(b, l, n, x, y, z, _dev=0):
        """
        Return the spherical harmonics of the vector (x,y,z)
        """
        R = SO3part.zeros(b, l, n)
        _SO3partB.view(R).add_spharm(x, y, z)
        if _dev > 0:
            return R.cuda()
        return R

    @staticmethod
    def Fzeros(b, l, _dev=0):
        """
        Create an SO(3)-part corresponding to the l'th matrix in the Fourier transform of a function on SO(3).
        This gives a b*(2+l+1)*(2l+1) dimensional complex tensor. 
        """
        if _dev == 0:
            return SO3part(torch.zeros([b, 2*l+1, 2*l+1, 2]))
        else:
            return SO3part(torch.zeros([b, 2*l+1, 2*l+1, 2])).cuda()

    @staticmethod
    def Frandn(b, l, _dev=0):
        """
        Create an SO(3)-part corresponding to the l'th matrix in the Fourier transform of a function on SO(3).
        This gives a b*(2+l+1)*(2l+1) dimensional complex random tensor. 
        """
        if _dev == 0:
            return SO3part(torch.randn([b, 2*l+1, 2*l+1, 2]))
        else:
            return SO3part(torch.randn([b, 2*l+1, 2*l+1, 2])).cuda()

    # ---- Access ------------------------------------------------------------------------------------------

    def getb(self):
        return self.size(2)

    def getl(self):
        return (self.size(1)-1)/2

    def getn(self):
        return self.size(3)

    # ---- Operations --------------------------------------------------------------------------------------

    def rotate(self, R):
        A = _SO3partB.view(self).apply(R)
        return SO3part(A.torch())

    def apply(self, R):
        return SO3part(_SO3partB.view(self).apply(R).torch())


    # ---- Products -----------------------------------------------------------------------------------------

    def CGproduct(self, y, l):
        """
        Compute the l component of the Clesbsch--Gordan product of this SO3part with another SO3part y.
        """
        return SO3part_CGproductFn.apply(self,y,l)

    #def DiagCGproduct(self, y, maxl=-1):
     #   """
      #  Compute the diagonal Clesbsch--Gordan product of this SO3vec with another SO3vec y.
       # """
        #r = SO3vec()
        #r.parts = list(SO3vec_DiagCGproductFn.apply(len(self.parts), len(y.parts), maxl, *(self.parts+y.parts)))
        #return r


    # ---- I/O ----------------------------------------------------------------------------------------------

    # def __str__(self):
    #    u=_SO3partB.view(self)
    #    return u.__str__()


# ----------------------------------------------------------------------------------------------------------
# ---- SO3vec ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class SO3vec:
    """
    An SO(3)-covariant vector consisting of a sequence of SO3part objects, each transforming according
    to a specific irrep of SO(3).
    """

    def __init__(self):
        self.parts = []

    def __init__(self,parts=None):
        self.parts=[]
        if parts is not None:
            for l in range(len(parts)):
                self.parts.append(parts[l])


    # ---- Static constructors ------------------------------------------------------------------------------

    @staticmethod
    def zeros(b, _tau, _dev=0):
        "Construct a zero SO3vec object of given type _tau."
        R = SO3vec()
        for l in range(0, len(_tau)):
            R.parts.append(SO3part.zeros(b, l, _tau[l], _dev))
        return R

    @staticmethod
    def randn(b, _tau, _dev=0):
        "Construct a random SO3vec object of given type _tau."
        R = SO3vec()
        for l in range(0, len(_tau)):
            R.parts.append(SO3part.randn(b, l, _tau[l], _dev))
        return R

    @staticmethod
    def spharm(b, _tau, x, y, z, _dev=0):
        """
        Compute a vector of spherical harmonic coefficients. 
        The values will be duplicated along the batch and channel dimensions.
        """
        R = SO3vec()
        for l in range(0, len(_tau)):
            R.parts.append(SO3part.spharM(b, l, _tau[l], x, y, z, _dev))
        return R

    @staticmethod
    def Fzeros(b, maxl, _dev=0):
        "Construct an SO3vec corresponding the to the Forier matrices 0,1,...maxl of b functions on SO(3)."
        R = SO3vec()
        for l in range(0, maxl+1):
            R.parts.append(SO3part.Fzeros(b, l, _dev))
        return R

    @staticmethod
    def Frandn(b, maxl, _dev=0):
        "Construct a zero SO3Fvec object  with l ranging from 0 to maxl."
        R = SO3vec()
        for l in range(0, maxl+1):
            R.parts.append(SO3part.Frandn(b, l, _dev))
        return R

    @staticmethod
    def zeros_like(x):
        R = SO3vec()
        # b=x.parts[0].dim(0)
        for l in range(0, len(x.parts)):
            R.parts.append(SO3part(torch.zeros_like(x.parts[l])))
        return R

    # ---- Access -------------------------------------------------------------------------------------------

    def tau(self):
        "Return the 'type' of the SO3vec, i.e., how many components it has corresponding to l=0,1,2,..."
        r = []
        for l in range(0, len(self.parts)):
            r.append(self.parts[l].size(2))
            # r.append(self.parts[l].getn())
        return r

    # ---- Transport ---------------------------------------------------------------------------------------

    def to(self, device):
        r = SO3vec()
        for p in self.parts:
            r.parts.append(p.to(device))
        return r

    # ---- Operations ---------------------------------------------------------------------------------------

    def rotate(self, R):
        "Apply the group element to this vector"
        r = SO3vec()
        for l in range(0, len(self.parts)):
            r.parts.append(self.parts[l].rotate(R))
        return r

    # ---- Products -----------------------------------------------------------------------------------------

    def CGproduct(self, y, maxl=-1):
        """
        Compute the full Clesbsch--Gordan product of this SO3vec with another SO3vec y.
        """
        r = SO3vec()
        r.parts = list(SO3vec_CGproductFn.apply(len(self.parts), len(y.parts), maxl, *(self.parts+y.parts)))
        return r

    def DiagCGproduct(self, y, maxl=-1):
        """
        Compute the diagonal Clesbsch--Gordan product of this SO3vec with another SO3vec y.
        """
        r = SO3vec()
        r.parts = list(SO3vec_DiagCGproductFn.apply(len(self.parts), len(y.parts), maxl, *(self.parts+y.parts)))
        return r

    def Fproduct(self, y, maxl=-1):
        """
        Compute the Fourier space product of this SO3Fvec with another SO3Fvec y.
        """
        r = SO3vec()
        r.parts = list(SO3vec_FproductFn.apply(len(self.parts), len(y.parts), maxl, *(self.parts+y.parts)))
        return r

    def Fmodsq(self, maxl=-1):
        """
        Compute the Fourier transform of the squared modulus of f.
        """
        r = SO3vec()
        r.parts = list(SO3vec_FmodsqFn.apply(len(self.parts), maxl, *(self.parts)))
        return r

    # ---- I/O ----------------------------------------------------------------------------------------------

    def __str__(self):
        u = _SO3vecB.view(self.parts)
        return u.__str__()


# ----------------------------------------------------------------------------------------------------------
# ---- Other functions --------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def CGproduct(x, y, maxl=-1):
    return x.CGproduct(y, maxl)


def DiagCGproduct(x, y, maxl=-1):
    return x.DiagCGproduct(y, maxl)


def Fproduct(x, y, a=-1):
    return x.Fproduct(y, a)


def Fmodsq(x, a=-1):
    return x.Fmodsq(a)


def tau_type(x):
    r = []
    for t in x:
        r.append(t.size(2))
    return r


def CGproductType(x, y, maxl=-1):
    if maxl == -1:
        maxl = len(x)+len(y)-2
    maxl = min(maxl, len(x)+len(y)-2)
    r = [0]*(maxl+1)
    for l1 in range(0, len(x)):
        for l2 in range(0, len(y)):
            for l in range(abs(l1-l2), min(l1+l2, maxl)+1):
                r[l] += x[l1]*y[l2]
    return r


def DiagCGproductType(x, y, maxl=-1):
    if maxl == -1:
        maxl = len(x)+len(y)-2
    maxl = min(maxl, len(x)+len(y)-2)
    r = [0]*(maxl+1)
    for l1 in range(0, len(x)):
        for l2 in range(0, len(y)):
            for l in range(abs(l1-l2), min(l1+l2, maxl)+1):
                r[l] += x[l1]
    return r


def MakeZeroSO3parts(b, tau, _dev=0):
    R = []
    for l in range(0, len(tau)):
        R.append(SO3part.zeros(b, l, tau[l], _dev))
    return R


def makeZeroSO3Fparts(b, maxl, _dev=0):
    R = []
    for l in range(0, maxl+1):
        R.append(SO3part.Fzeros(b, l, _dev))
        # R.append(torch.zeros(b,2*l+1,2*l+1,2,dtype=torch.float))
    return R


# ----------------------------------------------------------------------------------------------------------
# ---- Autograd functions -----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class SO3part_CGproductFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y,l):
        ctx.l=l
        ctx.save_for_backward(x,y)

        b = x.size(0)
        dev = int(x.is_cuda)
        r = SO3part.zeros(b,l,x.size(2)*y.size(2),dev)

        _x = _SO3partB.view(x)
        _y = _SO3partB.view(y)
        _r = _SO3partB.view(r)
        _r.addCGproduct(_x,_y)

        return r

    @staticmethod
    def backward(ctx, g):

        x,y = ctx.saved_tensors

        xg=torch.zeros_like(x)
        yg=torch.zeros_like(y)

        _x = _SO3partB.view(x)
        _y = _SO3partB.view(y)

        _g = _SO3partB.view(g)
        _xg = _SO3partB.view(xg)
        _yg = _SO3partB.view(yg)

        _xg.addCGproduct_back0(_g, _y)
        _yg.addCGproduct_back1(_g, _x)

        return xg,yg,None



class SO3vec_CGproductFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, k1, k2, maxl, *args):
        ctx.k1 = k1
        ctx.k2 = k2
        # ctx.maxl=maxl
        ctx.save_for_backward(*args)

        b = args[0].size(0)
        tau = CGproductType(tau_type(args[0:k1]), tau_type(args[k1:k1+k2]), maxl)
        dev = int(args[0].is_cuda)
        r = MakeZeroSO3parts(b, tau, dev)

        _x = _SO3vecB.view(args[0:k1])
        _y = _SO3vecB.view(args[k1:k1+k2])
        _r = _SO3vecB.view(r)
        _r.addCGproduct(_x, _y)

        return tuple(r)

    @staticmethod
    def backward(ctx, *args):

        k1 = ctx.k1
        k2 = ctx.k2
        # maxl=ctx.maxl

        inputs = ctx.saved_tensors
        assert len(inputs) == k1+k2, "Wrong number of saved tensors."

        grads = [None, None, None]
        for i in range(k1+k2):
            grads.append(torch.zeros_like(inputs[i]))

        _x = _SO3vecB.view(inputs[0:k1])
        _y = _SO3vecB.view(inputs[k1:k1+k2])

        _g = _SO3vecB.view(args)
        _xg = _SO3vecB.view(grads[3:k1+3])
        _yg = _SO3vecB.view(grads[k1+3:k1+k2+3])

        _xg.addCGproduct_back0(_g, _y)
        _yg.addCGproduct_back1(_g, _x)

        return tuple(grads)


class SO3vec_DiagCGproductFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, k1, k2, maxl, *args):
        ctx.k1 = k1
        ctx.k2 = k2
        # ctx.maxl=maxl
        ctx.save_for_backward(*args)

        b = args[0].size(0)
        tau = DiagCGproductType(tau_type(args[0:k1]), tau_type(args[k1:k1+k2]), maxl)
        dev = int(args[0].is_cuda)
        r = MakeZeroSO3parts(b, tau, dev)

        _x = _SO3vecB.view(args[0:k1])
        _y = _SO3vecB.view(args[k1:k1+k2])
        _r = _SO3vecB.view(r)
        _r.addDiagCGproduct(_x, _y)

        return tuple(r)

    @staticmethod
    def backward(ctx, *args):

        k1 = ctx.k1
        k2 = ctx.k2
        # maxl=ctx.maxl

        inputs = ctx.saved_tensors
        assert len(inputs) == k1+k2, "Wrong number of saved tensors."

        grads = [None, None, None]
        for i in range(k1+k2):
            grads.append(torch.zeros_like(inputs[i]))

        _x = _SO3vecB.view(inputs[0:k1])
        _y = _SO3vecB.view(inputs[k1:k1+k2])

        _g = _SO3vecB.view(args)
        _xg = _SO3vecB.view(grads[3:k1+3])
        _yg = _SO3vecB.view(grads[k1+3:k1+k2+3])

        _xg.addDiagCGproduct_back0(_g, _y)
        _yg.addDiagCGproduct_back1(_g, _x)

        return tuple(grads)


class SO3vec_FproductFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, k1, k2, _maxl, *args):
        ctx.k1 = k1
        ctx.k2 = k2
        ctx.save_for_backward(*args)

        b = args[0].size(0)
        if _maxl == -1:
            maxl = k1+k2-2
        else:
            maxl = _maxl
        dev = int(args[0].is_cuda)

        r = makeZeroSO3Fparts(b, maxl, dev)

        _x = _SO3Fvec.view(args[0:k1])
        _y = _SO3Fvec.view(args[k1:k1+k2])
        _r = _SO3Fvec.view(r)
        _r.addFproduct(_x, _y)

        return tuple(r)

    @staticmethod
    def backward(ctx, *args):

        k1 = ctx.k1
        k2 = ctx.k2
        # maxl=ctx.maxl

        inputs = ctx.saved_tensors
        assert len(inputs) == k1+k2, "Wrong number of saved tensors."

        grads = [None, None, None]
        for i in range(k1+k2):
            grads.append(torch.zeros_like(inputs[i]))

        _x = _SO3Fvec.view(inputs[0:k1])
        _y = _SO3Fvec.view(inputs[k1:k1+k2])

        _g = _SO3Fvec.view(args)
        _xg = _SO3Fvec.view(grads[3:k1+3])
        _yg = _SO3Fvec.view(grads[k1+3:k1+k2+3])

        _xg.addFproduct_back0(_g, _y)
        _yg.addFproduct_back1(_g, _x)

        return tuple(grads)


class SO3vec_FmodsqFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, k1, _maxl, *args):
        ctx.k1 = k1
        # ctx.k2=k1
        ctx.save_for_backward(*args)

        b = args[0].size(0)
        if _maxl == -1:
            maxl = k1+k1-2
        else:
            maxl = _maxl
        dev = int(args[0].is_cuda)

        r = makeZeroSO3Fparts(b, maxl, dev)

        _x = _SO3Fvec.view(args[0:k1])
        # _y=_SO3Fvec.view(args[k1:k1+k2]);
        _r = _SO3Fvec.view(r)
        _r.addFproduct(_x, _x)

        return tuple(r)

    @staticmethod
    def backward(ctx, *args):

        k1 = ctx.k1
        # k2=ctx.k2

        inputs = ctx.saved_tensors
        assert len(inputs) == k1, "Wrong number of saved tensors."

        grads = [None, None]
        for i in range(k1):
            grads.append(torch.zeros_like(inputs[i]))

        _x = _SO3Fvec.view(inputs[0:k1])
        # _y=_SO3Fvec.view(inputs[k1:k1+k2]);

        _g = _SO3Fvec.view(args)
        _xg = _SO3Fvec.view(grads[2:k1+2])
        # _yg=_SO3Fvec.view(grads[k1+3:k1+k2+3]);

        _xg.addFproduct_back0(_g, _x)
        _xg.addFproduct_back1(_g, _x)

        return tuple(grads)
