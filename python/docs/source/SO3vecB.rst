******
SO3vec
******


An ``SO3vec`` object represents a general SO(3)-covariant vector, stored 
as a combination of of ``SO3part`` s. 
All the ``SO3part`` s must have the same batch dimension. 

The `type` of an ``SO3vec`` is a dictionary specifying the number of fragments in each of its parts. 
For example, the following creates a random ``SO3vec`` of type (2,3,1). 

.. code-block:: python

  >> v=gelib.SO3vec.randn(1,{0:2,1:3,2:1})
  >> print(v.repr())
  <GElib::SO3vec of type (2,3,1)>
  >> print(v)
  Part l=0:
    [ (0.132629,0.950553) (0.719683,1.16923) ]
   
  
  Part l=1:
    [ (-0.308873,1.34239) (-0.0749153,0.787603) (0.124809,-0.68182) ]
    [ (-0.395814,-0.452225) (-0.301379,-0.498362) (0.368224,0.251531) ]
    [ (1.73902,-0.423323) (-0.411957,0.293598) (-1.11078,-0.537569) ]
  
  
  Part l=2:
    [ (0.295592,-0.0414616) (1.63098,0.730143) (-0.0242692,0.707672) (0.771041,-0.809959) (0.763403,0.260789) ]
  

The batch dimension and type of an SO(3)-vector can be accessed as follows.

.. code-block:: python

 >> v.getb()
 1
 >> v.tau()
 {0:2, 1:3, 2:1}

The invidual parts are stored in the ``parts`` member variable

.. code-block:: python

  >> print(v.parts[1])

  tensor([[[-0.3089+1.3424j, -0.0749+0.7876j,  0.1248-0.6818j],
           [-0.3958-0.4522j, -0.3014-0.4984j,  0.3682+0.2515j],
           [ 1.7390-0.4233j, -0.4120+0.2936j, -1.1108-0.5376j]]])

|

===============
Fourier vectors
===============

One context in which ``SO3vec`` objects appear is when computing the 
Fourier transform of functions on SO(3). In this case, the ``SO3part``\s have the same number of fragments 
(channels) as the dimensionality of the corresponding irreps. 
Such ``SO3vec`` objects are created with the ``Fzero`` or ``Frandn`` constructors, which take only two 
arguments: the batch dimension and :math:`\ell_{\textrm{max}}`. 

.. code-block:: python

 >> v=gelib.SO3vec.Frandn(1,2)
 >> print(v.repr())

 <GElib::SO3vec of type (1,3,5)>

 >> print(v)

 Part l=0:
   [ (1.87611,-0.890737) ]

 Part l=1:
   [ (0.13171,-1.274) (2.0406,1.17752) (-1.80254,-0.340302) ]
   [ (0.640126,-1.1053) (-0.785457,0.986579) (0.725949,0.430661) ]
   [ (-1.25901,1.26772) (1.22261,-0.704127) (-1.27295,0.0716574) ]


 Part l=2:
   [ (-0.699084,-1.68197) (-0.482411,-1.48628) (0.215704,1.25033) (0.551469,0.42062) (0.795124,0.636616) ]
   [ (0.522405,-1.62037) (0.479887,1.40499) (0.605501,0.366552) (-1.01028,-0.662143) (2.46867,0.250409) ]
   [ (0.0376103,1.33382) (-0.336708,0.671129) (-0.23257,-1.01927) (-1.10624,-0.912405) (-1.49729,-1.13004) ]
   [ (0.490532,0.364831) (1.62448,-0.31748) (-0.101089,-0.300246) (1.36258,-0.823076) (-1.61671,0.0582258) ]
   [ (0.443963,1.07747) (-1.57394,1.58904) (0.0186187,-0.376147) (0.970686,-0.55809) (0.39142,1.74658) ]

.. 
 In addition to all the operations that can be applied to generic ``SO3vec`` objects, Fourier ``SO3vec``\s 
 also support the ``Fproduct`` and ``Fmodsq`` operations. 

|

=======================
Clebsch-Gordan products
=======================

The full Clebsch-Gordan product (CG-product) of two SO3-vectors is computed as follows.

.. code-block:: python

  >> u=gelib.SO3vec.randn(1,{0:2,1:2})
  >> v=gelib.SO3vec.randn(1,{0:2,1:2})
  >> w=gelib.CGproduct(u,v)
  >> print(w)

  Part l=0:
    [ (0.152031,-0.140948) (-0.176707,0.0986708) (-0.0514539,2.16813) (0.54849,-2.04492) (-1.24255,-0.815015) (-1.40811,-0.123935) (-0.391867,1.13209) (-0.161307,-0.330928) ]

  Part l=1:
    [ (0.0961476,-0.243252) (0.171405,-0.405961) (1.1234,2.495) (1.79502,4.24597) (-0.730597,0.187905) (0.736381,-0.00987765) (0.698929,0.568218) (-0.532079,-0.700114) (-0.163401,0.429268) (-0.412671,1.27816) (0.850947,-1.12338) (2.10184,-2.1415) ]
    [ (-0.0326659,-0.024234) (0.00847598,0.172192) (0.419973,-0.0682939) (-1.35334,-1.19208) (-0.374269,0.472096) (0.463849,-0.361595) (1.51776,-0.805567) (-1.62546,0.414405) (-0.0409343,-0.262541) (-0.664351,-1.61683) (-0.958011,-0.645344) (-2.28508,0.289834) ]
    [ (0.304888,-0.0110071) (0.0900037,-0.295688) (-2.14074,2.36709) (1.5615,2.83128) (-0.456644,-0.978039) (0.207788,1.03305) (-0.936221,-0.103796) (0.864218,0.314231) (0.494024,-0.0305465) (0.703364,-0.464528) (-1.7338,-0.26607) (-0.553973,1.15706) ]


  Part l=2:
    [ (-0.728853,0.612083) (-1.2514,1.00255) (1.10502,0.265513) (1.85081,0.490198) ]
    [ (-0.0748801,0.618935) (-0.0294498,0.478025) (0.970257,-1.34299) (1.06154,-1.94166) ]
    [ (-0.750575,-0.508764) (-1.11883,-0.688738) (-0.534416,0.477625) (-0.479641,1.47743) ]
    [ (-0.737463,0.204984) (0.522857,0.792809) (1.75326,0.526698) (1.35542,-1.92182) ]
    [ (0.430848,-1.52889) (-1.32246,-0.916922) (-0.943746,-1.01531) (-1.28382,0.569196) ]

The optional third argument of ``CGproduct`` can be used to limit the result to parts 
:math:`\ell=0,1,\ldots,\ell_{\text{max}}`. 

.. code-block:: python

  >> w=gelib.CGproduct(u,v,1)
  >> print(w)

  Part l=0:
    [ (0.152031,-0.140948) (-0.176707,0.0986708) (-0.0514539,2.16813) (0.54849,-2.04492) (-1.24255,-0.815015) (-1.40811,-0.123935) (-0.391867,1.13209) (-0.161307,-0.330928) ]


  Part l=1:
    [ (0.0961476,-0.243252) (0.171405,-0.405961) (1.1234,2.495) (1.79502,4.24597) (-0.730597,0.187905) (0.736381,-0.00987765) (0.698929,0.568218) (-0.532079,-0.700114) (-0.163401,0.429268) (-0.412671,1.27816) (0.850947,-1.12338) (2.10184,-2.1415) ]
    [ (-0.0326659,-0.024234) (0.00847598,0.172192) (0.419973,-0.0682939) (-1.35334,-1.19208) (-0.374269,0.472096) (0.463849,-0.361595) (1.51776,-0.805567) (-1.62546,0.414405) (-0.0409343,-0.262541) (-0.664351,-1.61683) (-0.958011,-0.645344) (-2.28508,0.289834) ]
    [ (0.304888,-0.0110071) (0.0900037,-0.295688) (-2.14074,2.36709) (1.5615,2.83128) (-0.456644,-0.978039) (0.207788,1.03305) (-0.936221,-0.103796) (0.864218,0.314231) (0.494024,-0.0305465) (0.703364,-0.464528) (-1.7338,-0.26607) (-0.553973,1.15706) ]

|

================================
Diagonal Clebsch-Gordan products
================================

In the full CG-product, every fragment of ``u`` is multiplied with every fragment of ``v``.  
This can lead to output vectors with a very large numbers of fragments. In 
contrast, the ``DiagCGproduct`` function only computes the product between corresponding fragments. 
Naturally, this means that ``u`` and ``v`` must have the same type.

.. code-block:: python

  >> w=gelib.DiagCGproduct(u,v)
  >> print(w)

  Part l=0:
    [ (0.152031,-0.140948) (0.54849,-2.04492) (-1.24255,-0.815015) (-0.161307,-0.330928) ]


  Part l=1:
    [ (0.0961476,-0.243252) (1.79502,4.24597) (-0.730597,0.187905) (-0.532079,-0.700114) (-0.163401,0.429268) (2.10184,-2.1415) ]
    [ (-0.0326659,-0.024234) (-1.35334,-1.19208) (-0.374269,0.472096) (-1.62546,0.414405) (-0.0409343,-0.262541) (-2.28508,0.289834) ]
    [ (0.304888,-0.0110071) (1.5615,2.83128) (-0.456644,-0.978039) (0.864218,0.314231) (0.494024,-0.0305465) (-0.553973,1.15706) ]


  Part l=2:
    [ (-0.728853,0.612083) (1.85081,0.490198) ]
    [ (-0.0748801,0.618935) (1.06154,-1.94166) ]
    [ (-0.750575,-0.508764) (-0.479641,1.47743) ]
    [ (-0.737463,0.204984) (1.35542,-1.92182) ]
    [ (0.430848,-1.52889) (-1.28382,0.569196) ]

|


===================
Fproduct and Fmodsq
===================

If ``F`` and ``G`` are the Fourier transforms of two functions :math:`f,g\colon \textrm{SO}(3)\to\mathbb{C}` 
(represented as Fourier ``SO3vec`` objects),  
the Fourier transform of the product  :math:`h(R)=f(R)\,G(R)` can be computed directly from ``F`` and ``G`` 
using the formula 

.. math::
 H_\ell=\frac{1}{8\pi^2} \sum_{\ell_1} \sum_{\ell_2} 
 \frac{(2\ell_1 +1)(2\ell_2 +1)}{(2\ell +1)}~
 C_{\ell_1,\ell_2,\ell}^\dag (F_\ell \otimes G_\ell)\, C_{\ell_1,\ell_2,\ell}.

This operation is performed by the ``Fproduct`` function. 

``Fmodsq`` uses a similar formula to compute the Fourier transform of the squared modulus function 
:math:`h(R)=|f(R)|^2`. 

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

|

