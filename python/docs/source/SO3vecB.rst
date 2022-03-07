******
SO3vec
******


An ``SO3vec`` object represents a general SO(3)-covariant vector and is stored 
as a sequence of ``SO3part`` objects. Once again, ``SO3vec`` also has an additional batch dimension. 

The `type` of an ``SO3vec`` is a list specifying the multiplicity of each of its parts. 
For example, the following creates a random ``SO3vec`` of type (2,3,1). 

.. code-block:: python

  >>> v=gelib.SO3vec.randn(1,[2,3,1])
  >>> v
  <gelib_torchC.SO3vec object at 0x7f974337bb70>
  >>> print(v)
  Part l=0:
    [ (0.289356,1.54426) (-1.34567,1.53707) ]


  Part l=1:
    [ (-1.01328,-0.592903) (-1.11794,0.749696) (-0.734772,1.43901) ]
    [ (0.541032,1.58891) (-1.55468,-0.30842) (-0.937764,0.634763) ]
    [ (0.871103,-0.726917) (0.787215,-1.39109) (2.66829,0.85663) ]


  Part l=2:
    [ (-0.440433,-0.919034) (-0.700111,-0.901544) (1.29377,-0.482789) (-1.26476,-1.61195) (-1.42624,-0.967444) ]


The type of an SO(3)-vector can be accessed as follows.

.. code-block:: python

 >>> v.tau()
 [2, 3, 1]

The invidual parts are stored in the ``parts`` member variable

.. code-block:: python

  >>> print(v.parts[1])
    [ (-1.01328,-0.592903) (-1.11794,0.749696) (-0.734772,1.43901) ]
    [ (0.541032,1.58891) (-1.55468,-0.30842) (-0.937764,0.634763) ]
    [ (0.871103,-0.726917) (0.787215,-1.39109) (2.66829,0.85663) ]


=======================
Clebsch-Gordan products
=======================

The Clebsch-Gordan product of two SO3-vectors is computed as follows.

.. code-block:: python

 >>> u=gelib.SO3vec.randn(1,[1,1])
 >>> v=gelib.SO3vec.randn(1,[1,1])
 >>> w=gelib.CGproduct(u,v)
 >>> print(w)
 Part l=0:
   [ (0.800454,-2.72231) (0.387997,-2.21325) ]
 
 Part l=1:
   [ (-1.08378,0.166964) (-1.13947,1.02458) (-0.979756,-0.170846) ]
   [ (-3.14667,-0.020229) (-1.60544,-0.595765) (-0.658927,1.13758) ]
   [ (0.573493,3.50629) (0.609701,-0.290724) (-1.86063,-0.256204) ]
 
 Part l=2:
   [ (0.545523,0.23039) ]
   [ (1.0578,1.10345) ]
   [ (0.098245,0.754121) ]
   [ (1.15855,-0.537074) ]
   [ (-0.530323,0.658823) ]


==============
GPU operations
==============


``SO3vec`` objects can be can moved back and forth between the host (CPU) and the GPU 
the same way as ``SO3part`` objects. 

.. code-block:: python

  >>> A=gelib.SO3vec.randn(1,[2,3,1])
  >>> B=A.to(device='cuda') # Create a copy of A on the first GPU (GPU0)
  >>> C=B.to(device='cpu') # Move B back to the host 

Similarly to the ``SO3part`` case, operations between GPU-resident ``SO3vec`` s are executed  
on the GPU and the result is placed on the same device.  

