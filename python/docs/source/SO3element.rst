**********
SO3element
**********

An ``SO3element`` object represents a single group element :math:`R\in\mathrm{SO}(3)`, 
stored as a three dimensional orthogonal matrix with unit determinant. 
The ``identity`` constructor returns the identity element of the group:

.. code-block:: python

  >> e=gelib.SO3element.identity()
  >> print(e)

  SO3element([[1., 0., 0.],
              [0., 1., 0.],
              [0., 0., 1.]])

while the 'random' constructor returns a group element chosen unformly at random (with respect to 
the Haar measure):

.. code-block:: python

  >> R=gelib.SO3element.random()
  >> print(R)

  SO3element([[ 0.0708,  0.9071,  0.4150],
              [-0.8343, -0.1742,  0.5231],
              [ 0.5468, -0.3832,  0.7444]])

Group elements can also be initialized directly from single precision PyTorch tensors. 

Group elements can be multiplied together simply with the ``*`` operator:

.. code-block :: python

  >> R1=gelib.SO3element.random()
  >> R2=gelib.SO3element.random()
  >> print(R1*R2)

  SO3element([[ 0.4764,  0.8586, -0.1896],
              [-0.7106,  0.5029,  0.4920],
              [ 0.5178, -0.0996,  0.8497]])

The ``inv`` method returns the inverse of a group element. 
Given that the group elements of :math:`\mathrm{SO}(3)` are orthogonal matrices, 
this is just the transpose of :math:`R`: 

.. code-block :: python

  >> R=g.SO3element.random()
  >> Rinv=R.inv()
  >> print(R*Rinv)
  SO3element([[ 1.0000e+00,  1.7881e-07, -2.9802e-08],
              [ 1.7881e-07,  1.0000e+00,  1.7881e-07],
              [-2.9802e-08,  1.7881e-07,  1.0000e+00]])
