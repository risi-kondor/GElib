###########
SO3 classes
###########


The core classes in GElib for storing and manipulating SO(3)-equivariant vectors are 
``SO3part`` and ``SO3vec``: 

#. ``SO3part`` stores :math:`n` vectors that transform according to a specific irreducible representation :math:`D^{(\ell)}` of SO(3). 

#.  ``SO3vec`` stores a vector that transforms according to a generic representation of SO(3). 
    Since any representation decomposes into a sum of irreducibles, internally, the vector is  
    stored as a sequence of ``SO3part`` objects.

The library also provides the ``SO3element`` and ``SO3type`` helper classes. 


