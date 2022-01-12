import torch

class ctens(torch.Tensor):

    #def __new__(cls,T,_bsize=1):
     #   a=super.__new__(cls)
      #  a.bsize=_bsize
       # return a

    def __init__(self,_T):
        self=_T
        #self.bsize=_bsize


    ## ---- Static constructors


    @staticmethod
    def zeros(_dims,_bsize=1):
        a=ctens(torch.zeros([2]+_dims)) # TODO
        #a=ctens(torch.zeros([2]+_dims,requires_grad=True)) # TODO
        a.bsize=_bsize
        return a
    
    @staticmethod
    def randn(_dims,_bsize=1):
        a=ctens(torch.randn([2]+_dims))
        #a=ctens(torch.randn([2]+_dims,requires_grad=True))
        a.bsize=_bsize
        return a


    ## ---- Operations ---------------------------------------------------------------------------------------


    def matmul(self,y):
        assert self.dim()==3, "Must be a matrix"
        assert y.dim()==3, "Must be a ctens matrix"

        I=self.size(1)
        J=self.size(2)
        K=y.size(2)
        assert J==y.size(1), "Inner dimension mismatch"

        r=ctens.zeros([I,K])
        rr=r.select(0,0)
        rr=torch.matmul(self.select(0,0),y.select(0,0))-torch.matmul(self.select(0,1),y.select(0,1))
        ri=r.select(0,0)
        ri=torch.matmul(self.select(0,0),y.select(0,1))+torch.matmul(self.select(0,1),y.select(0,0))
        return r;

 
    def mix_blocks(self,M):
        "Mix blocks with each other by M"
        assert self.dim()==3, "Must be a matrix"
        assert M.dim()==3,"M must be a ctens matrix"

        nb=self.size(2)/self.bsize
        nnb=M.size(2)
        
        I=self.size(1)
        J=nb
        K=nnb
        assert J==M.size(1), "Inner dimension mismatch"
        
        r=ctens.zeros([I,nnb*bsize],bsize)
        for a in range(bsize):
            rr=ctens.select(0,0).reshape([I,nnb,self.bsize]).select(2,a)
            rr=ctens.select(0,1).reshape([I,nnb,self.bsize]).select(2,a)
            rr=(torch.matmul(self.select(0,0).reshape([I,nb,self.bsize]).select(2,a),y.select(0,0))-
                torch.matmul(self.select(0,1).reshape([I,nb,self.bsize]).select(2,a),y.select(0,1)))
            ri=(torch.matmul(self.select(0,0).reshape([I,nb,self.bsize]).select(2,a),y.select(0,1))+
                torch.matmul(self.select(0,1).reshape([I,nb,self.bsize]).select(2,a),y.select(0,0)))
        return r;


    def matmul_each_block(M):
        "Multiply each block by the same ctens matrix M"
        assert self.dim()==3, "Must be a matrix"
        assert M.dim()==3, "Must be a matrix"

        nb=self.size(2)/self.bsize
        I=self.size(1)*nb
        J=self.bsize
        K=M.size(2)
        assert J==M.size(1), "Inner dimension mismatch"

        r=ctens.zeros([self.size(1),nb*K],K)
        rr=r.select(0,0).reshape([I,K])
        ri=r.select(0,1).reshape([I,K])
        xr=self.select(0,0).reshape(I,J)
        xi=self.select(0,1).reshape(I,J)
        rr=torch.matmul(xr,M.select(0,0))-torch.matmul(xi,M.select(0,1))
        ri=torch.matmul(xr,M.select(0,1))+torch.matmul(xi,M.select(0,0))
        return r
            
        
    def matmul_each_block_by_corresponding_block(M):
        "Multiply each block by the corresponding block in MM"
        assert self.dim()==3, "Must be a matrix"
        assert M.dim()==3, "Must be a matrix"

        nb=self.size(2)/self.bsize
        I=self.size(1)
        J=self.bsize
        K=M.size(2)
        assert nb==M.size(1)/m.bsize, "Mismatch in number of blocks"
        assert J==M.bsize, "Inner dimension mismatch"

        r=ctens.zeros([I,nb*K],K)
        for a in range(nb):
            rr=r.select(0,0).reshape([I,nb,K]).select(1,a)
            ri=r.select(0,1).reshape([I,nb,K]).select(1,a)
            xr=self.select(0,0).reshape([I,nb,J]).select(1,a)
            xi=self.select(0,1).reshape([I,nb,J]).select(1,a)
            Mr=M.select(0,0).reshape([nb,J,K]).select(0,a)
            Mr=M.select(0,1).reshape([nb,J,K]).select(0,a)
            rr=torch.matmul(xr,Mr)-torch.matmul(xi,Mi)
            ri=torch.matmul(xr,Mi)+torch.matmul(xi,Mr)
        return r
            
        
    def __mul__(self,y):
        return self.matmul(y)


    ## ---- I/O ---------------------------------------------------------------------------------------------


    def __str__(self):
        return str(torch.Tensor(self))
        #return super().__str__()

    def __repr__(self):
        return "<ctens("+str(self.size())+")>"



## ---- ctens_list ------------------------------------------------------------------------------------------


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

    
