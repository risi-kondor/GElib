###########
SO3 classes
###########


The fundamental classes in GElib for storing and manipulating SO(3)-equivariant vectors are ``SO3vec``, 
``SO3part`` and ``SO3type``. 

``SO3vec`` stores a vector that the group SO(3) acts on. 
Any such vector can be decomposed into a sequence of smaller vectors that :math:`\mathrm{SO}(3)` 
acts on in an irreducible way, i.e., vectors that transform as 
:math:`\mathbf{v}^{(\ell)}_i\mapsto D^{(l)}(R)\,\mathbf{v}^{(\ell)}_i` for one of the irreducible 
representations :math:`D^{(l)}(R)` of :math:`\mathrm{SO}(3)`. 

``SO3part`` collects those components of an SO3-vector that transform according to the *same* irreducible 
representation. 
Finally, ``SO3type`` is a vector of integers that specifies how many irreducible components corresponding to 
:math:`D^{(0)},D^{(1)},\ldots` are featured in a given SO3-vector. 

