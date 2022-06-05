*******
SO3part
*******

An SO3-part :math:`P` is a collection of :math:`n` different complex valued vectors, 
which we call `fragments`, transforming according 
to the same irreducible representation :math:`D^{(\ell)}` of SO(3). 
The irreducible representations of SO(3) are sometimes referred to as Wigner D-matrices.  

Since :math:`D^{(\ell)}` is a :math:`2\ell\!+\!1` a dimensional representation, 
:math:`P` can be thought of as a matrix :math:`\mathbb{C}^{(2\ell+1)\times n}`.  
However, to aid in parallelizing certain operations, the  
``SO3part`` class in GElib also admits a "batch dimension", :math:`b`.  
Therefore, ``SO3part`` is implemented as a  
``(b,2l+1,n,2)`` -dimensional single precision ``torch.Tensor`` object,  
where the last dimension corresponds to the real/imaginary parts. 
This format may be converted to PyTorch's new complex tensor format with  
``torch.view_as_complex()``, while ``torch.view_as_real()`` converts in the opposite direction.

The following code constructs an ``SO3part`` object holding ``n=3`` Gaussian distributed random vectors corresponding 
to the ``l=2`` irrep of SO(3) (batch size 1) and prints it out in GElib's own compact format.  

.. code-block:: python

 >>> P=gelib.SO3part.randn(1,2,3)
 >>> print(P)
 [ (-0.254751,1.06977) (-0.0920261,0.568582) (0.61119,0.145976) ]
 [ (-2.08142,-1.54979) (1.86277,0.739985) (-0.347901,-1.85592) ]
 [ (0.537095,-1.018) (0.164219,-0.0593679) (-0.238732,0.50693) ]
 [ (1.17258,-0.634835) (-0.649356,-0.0877365) (1.00715,-0.304671) ]
 [ (-1.8357,-1.70148) (-0.851959,-1.31661) (0.824596,-0.197399) ]

A zero ``SO3part`` would be constructed similarly using the ``gelib.SO3part.zeros(b,l,n)`` constructor. 
In both cases the optional ``dev`` argument selects whether the ``SO3part`` is  
placed on the CPU (``dev=0``) or the GPU (``dev=1``). 
The batch dimension, ``b``, order ``l``, and multiplicity ``n`` of the ``SO3part`` are read out as follows.

.. code-block:: python

 >>> P.getb()
 1
 >>> P.getl()
 2
 >>> P.getn()
 3
 
Since ``SO3part`` is derived from ``torch.Tensor``, it supports all the usual  
tensor arithmetic operations, and its elements can be 
accessed directly, just like any PyTorch tensor. 

|

===================
Spherical harmonics
===================


``SO3part`` objects can also be initialized as the spherical harmonic coefficients 
(for a given value of :math:`\ell`) 
of corresponding three dimensional Euclidean vectors.  
The following example computes the :math:`\ell=1` spherical harmonics of a collection 
of random vectors with b=2 and n=4.

.. code-block:: python

 >>> A=torch.randn([2,3,4])
 >>> P=gelib.SO3part.spharm(1,A)
 >>> print(P)
 [ (0.0781101,0.0435727) (-0.261583,-0.170353) (-0.193139,-0.0910122) (0.230589,-0.236448) ]
 [ (-0.471946,0) (-0.209383,0) (-0.384136,0) (-0.143436,0) ]
 [ (-0.0781101,0.0435727) (0.261583,-0.170353) (0.193139,-0.0910122) (-0.230589,-0.236448) ]

 [ (-0.00622543,0.0919017) (-0.0485813,-0.340067) (0.21314,0.26868) (0.196154,-0.284294) ]
 [ (-0.470917,0) (0.0521697,0) (-0.0591338,0) (-0.0115272,0) ]
 [ (0.00622543,0.0919017) (0.0485813,-0.340067) (-0.21314,0.26868) (-0.196154,-0.284294) ]

|

============
Group action
============

The group SO(3) acts on :math:`P` by :math:`P\mapsto D^{(l)}(r) P`.  
This is implemented in the ``rotate`` method.  

.. code-block:: python

  >>> r=gelib.SO3element.uniform()
  >>> Pr=P.rotate(r)
  >>> print(Pr)
  [ (-1.11709,0.147657) (-1.11769,1.24745) ]
  [ (-1.60924,0.32233) (1.23518,1.00509) ]
  [ (0.618006,-0.476802) (-0.583183,-0.775842) ]
  [ (2.06776,-0.517233) (0.637578,0.790121) ]
  [ (-1.09639,0.00411239) (0.878735,-1.1598) ]

|

=======================
Clebsch-Gordan products
=======================

The :math:`l`'th component of the Clebsch-Gordan product of two SO3-vectors is computed as follows.

.. code-block:: python

  >>> P=gelib.SO3part.randn(1,1,2)
  >>> Q=gelib.SO3part.randn(1,2,2)
  >>> R=gelib.CGproduct(P,Q,1)
  >>> print(R)
  [ (-1.48122,-0.946409) (-1.19139,-2.46886) (-1.59038,2.91211) (-2.47551,1.86631) ]
  [ (-0.702674,-2.68819) (1.58708,-1.76738) (-0.247673,2.48612) (-1.36402,0.677406) ]
  [ (-0.752623,-0.900017) (-0.366799,1.73799) (0.0230675,0.234838) (-1.811,-1.47884) ]

``CGproduct`` and its following variants are implemented as differentiable operations, 
so GElib can propagate gradients back through them. 

.. note::
  The CG-product of two SO3parts is essentially a tensor product followed by a fixed linear transformation. 
  Since this operation is critical to certain types of equivariant neural networks, 
  GElib uses highly optimized routines for computing the CG-product on both the CPU and the GPU. 

  First, the tensor product is never explicitly formed, potentially saving significant amounts of 
  memory in neural network applications, where the results of intermediate calculations generally need to be 
  saved for the backward pass. 

  Second, the linear transformation has a specific sparsity pattern, whereby 
  :math:`{}_{[C_{\ell_1,\ell_2}^\ell]_{m_1,m_2,m}=0}` unless :math:`m_1+m_2=m`. 
  GElib uses specialized multiplication kernels for the CG-product that 
  exploit this symmetry. 

  Finally, the coefficients of the transformation, the so-called CG-coefficents, 
  are computed once and then cached separately on both the CPU and the GPU. In the case of the latter, 
  to the extent possible, GElib stores the coefficients in so-called `constant memory`, 
  which makes it possible to broadcast the coefficients to multiple streaming multiprocessors fast.
 

| 

================================
Diagonal Clebsch-Gordan products
================================

In the full CG-product, every fragment of ``P`` is multiplied with every fragment of ``Q``.  
In contrast, the ``DiagCGproduct`` function only computes the product between corresponding fragments. 

.. code-block:: python

  >>> R=gelib.DiagCGproduct(P,Q,1)
  >>> print(R)
  [ (-1.48122,-0.946409) (-2.47551,1.86631) ]
  [ (-0.702674,-2.68819) (-1.36402,0.677406) ]
  [ (-0.752623,-0.900017) (-1.811,-1.47884) ]

| 



==============
GPU operations
==============

``SO3part`` objects can be moved to the GPU or moved back to the host (CPU) just like any PyTorch tensor. 

.. code-block:: python

  >>> B=A.to(device='cuda') # Create a copy of A on the first GPU (GPU0)
  >>> C=B.to(device='cpu') # Move B back to the host 

In general, when all operands of a given operation are on the GPU, the computation is performed on 
the GPU and the result placed on the same GPU. 

|

