The **Clebsch--Gordan product** is a closely related operation on G-covariant parts. 
In particular, defining :math:`C_{a,b}^{i,j}` as the block of rows of :math:`C_{a,b}` corresponding 
to the :math:`j`\'th :math:`\rho_i(x)` block in :eq:`CGdecomp`, given a pair of G-covariant parts 
:math:`F_a` and :math:`F_b` corresponding to :math:`\rho_a` resp. :math:`\rho_b`, their 
Clebsch--Gordan product is the sequence of matrices 

.. math::
   F_a\otimes_{i,j} F_b=C_{a,b}^{i,j}(F_a\otimes F_b). 



Let us now split :math:`C_{a,b}` into blocks of rows :math:`(C_{a,b}^{i,j})` corresponding to the individual 
:math:`\rho_i(x)` blocks appearing in :eq:`CGdecomp`. 



Plugging in :eq:`CGdecomp`, and multiplying both sides of this equation by :math:`C_{a,b}`, 
using the unitarity of this matrix, we get 

.. math::
  C_{a,b} (F_a\otimes F_b) \longmapsto \bigg[\bigoplus_i \bigoplus_{j=1}^{\kappa(a,b,i)} \rho_i(x) \bigg] C_{a,b} (F_a\!\otimes\! F_b).  

Breaking this equation down according to the block structure of the matrix in square brackets, 
in particular, letting :math:`C_{a,b}^{i,j}` be the block of :math:`C_{a,b}` corresponding 
to the :math:`j`\'th 
:math:`\rho_i(x)` block, 
