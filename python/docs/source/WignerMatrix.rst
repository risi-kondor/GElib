***************
Wigner matrices
***************

The irreducible representations (irreps) of SO(3) are matrix valued functions 

.. math::
  D^{(\ell)}\colon \mathrm{SO}(3)\to\mathbb{C}^{(2\ell+1)\times(2\ell+1)} \hspace{100pt} \ell=0,1,2,\ldots 

sometimes also referred to as "Wigner D-matrices". 
Thus, the irrep indexed by :math:`\ell` is :math:`2\ell+1` dimensional. 

The ``WignerMatrix(l,phi,theta,psi)`` function returns the matrix of the :math:`\ell`\'th irrep 
corresponding to the Euler angles :math:`(\phi,\theta,\psi)`. 

.. code-block:: python

 >>> A=gelib.WignerMatrix(2,1.0,1.0,1.0)
 >>> A
 tensor([[-0.3877-4.4888e-01j, -0.6416+9.1454e-02j, -0.1804+3.9428e-01j, 0.1045+1.6275e-01j,  0.0528+0.0000e+00j],
         [ 0.6416-9.1454e-02j, -0.0258+5.6447e-02j,  0.3009+4.6856e-01j, 0.4782-1.4901e-08j,  0.1045-1.6275e-01j],
         [-0.1804+3.9428e-01j, -0.3009-4.6856e-01j, -0.0621+0.0000e+00j, 0.3009-4.6856e-01j, -0.1804-3.9428e-01j],
         [-0.1045-1.6275e-01j,  0.4782+1.4901e-08j, -0.3009+4.6856e-01j, -0.0258-5.6447e-02j, -0.6416-9.1454e-02j],
         [ 0.0528+0.0000e+00j, -0.1045+1.6275e-01j, -0.1804-3.9428e-01j, 0.6416+9.1454e-02j, -0.3877+4.4888e-01j]])

|


