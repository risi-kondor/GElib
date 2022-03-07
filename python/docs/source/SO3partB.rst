*******
SO3part
*******

An ``SO3part`` is a collection of :math:`n` different complex valued vectors transforming according to the same 
irreducible representation :math:`D^{(\ell)}` of SO(3). 
Given that :math:`D^{(\ell)}` is a :math:`2\ell\!+\!1` dimensional representation, this would 
correspond to a complex matrix :math:`A\in\mathbb{C}^{(2\ell+1)\times n}`.  
However, following PyTorch convention, ``SO3part`` objects can also have a "batch dimension" 
the significance of which is that GPU operations are automatically parallelized across this dimension. 
  
``SO3part`` is thus implemented as a  
:math:`b\times (2\ell+1)\times n\times 2` dimensional single precision 
PyTorch tensor, where ``b`` is the batch dimension and the last dimension corresponds to the real/imaginary parts. 
This format may be converted to PyTorch's new complex tensor format with  
``torch.view_as_complex()``, while ``torch.view_as_real()`` converts in the opposite direction.

The following code constructs an ``SO3part`` object holding ``n=3`` Gaussian distributed random vectors corresponding 
to the ``l=2`` irrep of SO(3) (batch size 1) and prints it out in GElib's own compact format.  

.. code-block:: python

 >>> A=gelib.SO3part.randn(1,2,3)
 >>> print(A)
 [ (-0.254751,1.06977) (-0.0920261,0.568582) (0.61119,0.145976) ]
 [ (-2.08142,-1.54979) (1.86277,0.739985) (-0.347901,-1.85592) ]
 [ (0.537095,-1.018) (0.164219,-0.0593679) (-0.238732,0.50693) ]
 [ (1.17258,-0.634835) (-0.649356,-0.0877365) (1.00715,-0.304671) ]
 [ (-1.8357,-1.70148) (-0.851959,-1.31661) (0.824596,-0.197399) ]

A zero ``SO3part`` would be constructed similarly using the ``gelib.SO3part.zeros(b,l,n)`` constructor. 
In both cases the optional ``dev`` argument selects whether the ``SO3part`` is  
placed on the CPU (``dev=0``) or the GPU (``dev=1``). 
The order ``l`` and multiplicity ``n`` of an ``A`` are read out as follows.

.. code-block:: python

 >>> A.getl()
 2
 >>> A.getn()
 3
 
Since ``SO3part`` is derived from ``torch.Tensor``, it supports all the usual  
arithmetic operations that can be applied to tensors.  

============
Group action
============

The group SO(3) acts on an ``SO3part`` :math:`A` by :math:`A\mapsto D^{(l)}(r) A` 
for any :math:`r\in\mathrm{SO}(3)`. This is implemented in the ``apply`` method.  

.. code-block:: python

  >>> r=gelib.SO3element.uniform()
  >>> Ar=A.apply(r)
  >>> print(Ar)
  [ (-1.11709,0.147657) (-1.11769,1.24745) ]
  [ (-1.60924,0.32233) (1.23518,1.00509) ]
  [ (0.618006,-0.476802) (-0.583183,-0.775842) ]
  [ (2.06776,-0.517233) (0.637578,0.790121) ]
  [ (-1.09639,0.00411239) (0.878735,-1.1598) ]

===================
Spherical harmonics
===================


``SO3part`` objects can be initialized as spherical harmonic coefficients of a three dimensional vector 
corresponding to a given :math:`\ell`.  

.. code-block:: python

 >>> A=gelib.SO3part.spharm(2,[0.12,0.31,-0.55])
 >>> print(A)
 [ (-0.0764131,-0.0695855) ]
 [ (-0.123458,0.318933) ]
 [ (0.37763,0) ]
 [ (0.123458,0.318933) ]
 [ (-0.0764131,0.0695855) ]


==============
GPU operations
==============

``SO3part`` objects can be moved to the GPU or moved back to the host (CPU) just like any PyTorch tensor. 

.. code-block:: python

  >>> B=A.to(device='cuda') # Create a copy of A on the first GPU (GPU0)
  >>> C=B.to(device='cpu') # Move B back to the host 

In general, when all operands of a given operation are on the GPU, the computation will be performed on 
the GPU and the result placed on the same GPU. 