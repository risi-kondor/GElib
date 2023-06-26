************************
Clebsch--Gordan matrices
************************

Given two irreps :math:`D^{(\ell_1)}` and :math:`D^{(\ell_2)}` of SO(3), the tensor product 
representation :math:`D^{(\ell_1)}\otimes D^{(\ell_2)}` reduces into a direct sum of irreps in the form 

.. math::

  (D^{(\ell_1)}\otimes D^{(\ell_2)})(R)= C_{\ell_1,\ell_2} \biggl[\bigoplus_{\ell=|\ell_1-\ell_2|}^{\ell_1+\ell_2} D^{(\ell)(R)}\biggr] C_{\ell_1,\ell_2}^\dag 

called the Clebsch--Gordan decomposition. 
The submatrix of :math:`C_{\ell_1,\ell_2}` corresponding to a particular value of :math:`\ell` we will denote 
:math:`C_{\ell_1,\ell_2,\ell}`. 

Taking the Clebsch--Gordan product of two (or more) ``SO3vec``\s is an equivariant operation and is an 
important building block of group equivariant neural networks.

The function ``SO3ClebschGordanTensor(l1,l2,l)`` returns :math:`C_{\ell_1,\ell_2,\ell}` as a third order tensor. 

.. code-block:: python

 >>> C=gelib.SO3CGtensor(1,2,2)
 >>> C
 tensor([[[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
	  [-0.5774,  0.0000,  0.0000,  0.0000,  0.0000],
	  [ 0.0000, -0.7071,  0.0000,  0.0000,  0.0000],
	  [ 0.0000,  0.0000, -0.7071,  0.0000,  0.0000],
	  [ 0.0000,  0.0000,  0.0000, -0.5774,  0.0000]],

	 [[ 0.8165,  0.0000,  0.0000,  0.0000,  0.0000],
	  [ 0.0000,  0.4082,  0.0000,  0.0000,  0.0000],
	  [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
	  [ 0.0000,  0.0000,  0.0000, -0.4082,  0.0000],
	  [ 0.0000,  0.0000,  0.0000,  0.0000, -0.8165]],

	 [[ 0.0000,  0.5774,  0.0000,  0.0000,  0.0000],
	  [ 0.0000,  0.0000,  0.7071,  0.0000,  0.0000],
	  [ 0.0000,  0.0000,  0.0000,  0.7071,  0.0000],
	  [ 0.0000,  0.0000,  0.0000,  0.0000,  0.5774],
	  [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]]])

As apparent from this output, the SO(3) Clebsch--Gordan coefficients satisfy
:math:`C_{\ell_1,\ell_2,\ell}(m_1,m_2,m)=0` unless :math:`m_1+m_2=m`. 
Hence, the coefficients can also given as elements of a :math:`(2\ell_1+1)\times (2\ell_2+1)` dimensional tensor.

.. code-block:: python

 >>> D=gelib.SO3CGmatrix(1,2,2)
 >>> D
 tensor([[ 0.0000, -0.5774, -0.7071, -0.7071, -0.5774],
         [ 0.8165,  0.4082,  0.0000, -0.4082, -0.8165],
         [ 0.5774,  0.7071,  0.7071,  0.5774,  0.0000]])
 