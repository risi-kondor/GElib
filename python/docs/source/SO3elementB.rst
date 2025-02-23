**********
SO3element
**********

An ``SO3element`` object represents a single group element :math:`R\in\mathrm{SO}(3)`. 
The group element can be initialized from its :math:`(\phi,\theta,\psi)` Euler angles or
chosen randomly from the group.

.. code-block:: python

 >>> a=gelib.SO3element(0.2,1.0,-0.1)
 >>> a
 SO3element(0.200000,1.000000,-0.100000)
 >>> b=gelib.SO3element.uniform()
 >>> b
 SO3element(0.534275,2.801079,1.191856) 

.. 
  The ``rho`` method returns the representation matrix (Wigner D-matrix) :math:`D^{(l)}(R)`.  
  Since :math:`D^{(l)}(R)` is a :math:`2\ell+1` dimensional representation, the result is a 
  :math:`(2\ell+1)\times(2\ell+1)` dimensional complex matrix. 

  .. code-block:: python

   >>> a=b.rho(2)
   >>> print(a)
   [ (-0.000784713,-0.000251944) (-0.00610006,0.00739689) (0.0328803,0.0598668) (0.32192,-0.0398973) (0.238549,-0.91275) ]
   [ (0.00934904,-0.00212605) (0.0128144,-0.081831) (-0.331814,-0.196326) (-0.680473,0.525486) (-0.0892214,-0.311872) ]
   [ (-0.0496073,0.0469496) (0.142627,0.358192) (0.832695,0) (-0.142627,0.358192) (-0.0496073,-0.0469496) ]
   [ (0.0892214,-0.311872) (-0.680473,-0.525486) (0.331814,-0.196326) (0.0128144,0.081831) (-0.00934904,-0.00212605) ]
   [ (0.238549,0.91275) (-0.32192,-0.0398973) (0.0328803,-0.0598668) (0.00610006,0.00739689) (-0.000784713,0.000251944) ]

