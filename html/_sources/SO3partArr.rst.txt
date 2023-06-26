************
SO3partArr
************

An ``SO3partArr`` object is an array of :math:`N` ``SO3part``\s, called cells, 
stored in a single tensor. 
The following constructs an ``SO3partArr`` object with batch dimension 1, 
consisting of :math:`2\times 2` cells, each 
holding ``n=3`` Gaussian distributed random vectors corresponding 
to the ``l=2`` irrep of SO(3) and prints it out.  

.. code-block:: python

 >>> A=gelib.SO3partArr.randn(1,[2,2],2,3)
 >>> print(A)
 Cell(0,0,0)
 [ (0.311398,-1.80913) (-0.921517,1.3563) (-0.463656,-0.153738) ]
 [ (-0.107248,-0.91566) (-0.268635,-0.934234) (0.292521,-0.466788) ]
 [ (2.03572,-0.917957) (0.231427,-0.438412) (-0.360624,0.0279882) ]
 [ (0.115751,1.16965) (0.588774,0.845822) (-0.291249,0.947408) ]
 [ (-0.0814305,0.592962) (-0.761416,0.177536) (-1.1615,-0.492729) ]

 Cell(0,0,1)
 [ (2.35398,0.425596) (-0.194245,0.566688) (1.38338,-2.13741) ]
 [ (-1.06957,-0.526105) (-0.759776,1.04892) (-1.01595,0.392939) ]
 [ (-0.581912,-1.07034) (-1.22874,1.92764) (-2.87221,-0.404009) ]
 [ (-0.609044,1.48928) (-0.236781,1.73229) (0.0744118,1.05315) ]
 [ (0.0746617,-0.272857) (0.225767,1.6043) (0.509717,1.18577) ]

 Cell(0,1,0)
 [ (0.947831,-3.48871) (-0.0304763,-2.04327) (0.564578,-1.22349) ]
 [ (-0.0346214,-1.32766) (-2.03267,-2.02395) (1.6006,1.92112) ]
 [ (-0.115995,-1.38828) (-1.59841,0.899515) (-1.56611,1.03086) ]
 [ (-1.72669,-1.18976) (-1.35317,0.0381025) (0.993481,-0.300125) ]
 [ (-1.89355,-0.901076) (-0.678269,-0.57792) (-1.81518,0.037866) ]

 Cell(0,1,1)
 [ (-1.29319,-1.78938) (-0.129255,-0.423372) (1.28934,-1.46638) ]
 [ (-1.03102,-0.603144) (0.0982916,-0.0189021) (0.0997176,-0.0518933) ]
 [ (1.18114,-0.145281) (1.03506,-1.23571) (-0.149596,-0.473458) ]
 [ (-1.46849,1.28294) (2.07751,-1.02977) (1.08815,0.0456315) ]
 [ (-1.47393,0.276098) (1.88762,0.878272) (-2.43639,1.74515) ]


An ``SO3partArr`` is stored as an :math:`b\times a_1\times \ldots\times a_k\times (2\ell+1)\times n` 
dimensional complex PyTorch tensor and supports the same operations as ``SO3part``, with the 
natural syntax. 

.. code-block:: python

 >>> A=gelib.SO3partArr.randn(1,[2,2],2,2)
 >>> B=gelib.CGproduct(A,A,2)
 >>> print(B)
 Cell(0,0,0)
 [ (-3.68829,2.1156) (0.70249,0.237766) (0.70249,0.237766) (0.779456,-1.94687) ]
 [ (-1.23101,-2.19237) (-1.71898,1.0264) (-1.71898,1.0264) (0.292517,2.35244) ]
 [ (-3.84478,1.74094) (-0.230101,0.727197) (-0.230101,0.727197) (-1.10904,2.02747) ]
 [ (-1.07394,-0.727521) (-0.295923,1.08078) (-0.295923,1.08078) (0.491645,-0.325617) ]
 [ (2.41142,-1.13035) (-0.65661,-0.304086) (-0.65661,-0.304086) (-1.05191,1.62797) ]

 Cell(0,0,1)
 [ (1.36789,-0.482399) (0.26131,-0.382551) (0.26131,-0.382551) (1.30767,-0.258516) ]
 [ (-2.16126,-0.749785) (0.0529939,1.20392) (0.0529938,1.20392) (0.122059,-2.28648) ]
 [ (-0.336205,1.51531) (0.736288,0.452221) (0.736288,0.452221) (0.804226,0.625281) ]
 [ (-0.160749,0.499992) (-0.0490017,0.55788) (-0.0490018,0.55788) (-1.2649,0.562428) ]
 [ (1.16493,0.181675) (0.53126,-0.995217) (0.53126,-0.995217) (1.13421,1.01864) ]

 Cell(0,1,0)
 [ (1.79327,0.325623) (0.0509059,0.148115) (0.0509059,0.148115) (0.22462,1.42646) ]
 [ (-3.50149,2.00715) (0.240442,-2.60512) (0.240442,-2.60512) (1.87925,1.41955) ]
 [ (-1.0096,-1.02473) (1.18226,-1.02291) (1.18226,-1.02291) (-0.576821,0.960422) ]
 [ (0.881412,1.7066) (-0.399855,0.0449982) (-0.399855,0.0449983) (-0.419007,-0.498288) ]
 [ (-2.69044,-2.11147) (2.12463,0.593042) (2.12463,0.593042) (-1.49771,0.35531) ]

 Cell(0,1,1)
 [ (-1.4145,3.08397) (2.16428,-0.107854) (2.16428,-0.107854) (-0.665168,-1.2601) ]
 [ (-3.04274,-3.63055) (-1.96958,3.30893) (-1.96958,3.30893) (3.23503,0.646061) ]
 [ (1.77954,-2.04437) (-1.35374,0.407467) (-1.35374,0.407467) (1.30007,-0.173447) ]
 [ (1.09669,1.0829) (0.786561,-0.675544) (0.786561,-0.675544) (-0.809771,-0.891165) ]
 [ (-1.0491,-0.0903285) (0.201155,2.46153) (0.201155,2.46153) (2.47398,-0.676472) ]

.. note::

 In some ways, the array dimensions of ``SO3partArr`` can be regarded as multiple batch indices. 
 Indeed, to compute cellwise operations like ``CGproduct`` on ``SO3partArr`` objects, 
 `GElib` reuses the same computational kernels as it uses for ``SO3part``, by internally concatenating 
 all the array dimensions into a single batch dimension. 

The main advantage of ``SO3partArr`` over ``SO3part`` is that it can take advantage of  
`cnine`'s cell operator functionality to perform efficient reductions and broadcasting 
along the array dimensions. 
The simplest of these operations is ``gather``. 

======
Gather
======

Given an ``SO3partArr`` object ``X`` with array dimension :math:`N` and an :math:`N\times N` 
real matrix :math:`C`, ``gather`` return an ``SO3partArr`` in which the :math:`i`'th cell is 

.. math:: Y^{(i)}=\sum_{j\::\:c_{i,j}\neq 0} c_{i,j}\,X^{(j)}. 

Using cnine's array functionality, GElib can efficiently parallelize this operation on the GPU, 
even when the number of terms in the sum for different values of :math:`i` is different. 

Invoking this functionality requires defining ``C`` as a single precision PyTorch tensor 
and constructing the corresponding ``cnine.Rmask1`` object:

.. code-block:: python

 >>> import cnine
 >>> C=torch.tensor([[0,1,0],[0,0,0],[1,0,0]],dtype=torch.float32);
 >>> mask=cnine.Rmask1(C)

The gather operation is then called as 

.. code-block:: python

 >>> X=gelib.SO3partArr.randn(1,[3],1,1,2)
 >>> Y=X.gather(mask)

Naturally, ``gather`` is a differentiable operation.


|


 