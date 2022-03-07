###########
SO3 classes
###########


The basic classes in GElib for storing and manipulating SO(3)-equivariant vectors are 
``SO3part`` and ``SO3vec``: 

#. An ``SO3part`` object represents :math:`n` vectors that transform according to a specific irreducible 
   representation :math:`D^{(\ell)}` of SO(3). 

#.  An ``SO3vec`` object represents a vector that transforms according to a general representation of SO(3). 
    The vector is stored as a collection of ``SO3part`` objects. 

``SO3part`` and ``SO3vec`` objects are complex valued. To facilitate parallelism, they also have an outer "batch" index 
running from 1 to :math:`b`. Consequently:

#.  An ``SO3part`` is stored as a :math:`b\times (2\ell+1)\times n\times 2` dimensional single precision 
    real ``torch.Tensor``. 

#.  An ``SO3vec`` is a collection of such tensors.

As usual, ``SO3part`` tensors can be converted to PyTorch's new complex tensor format using 
``torch.view_as_complex()``, while the reverse conversion is done with  
``torch.view_as_real()``.

|

The library also provides arrayed versions of these classes:

#.  An ``SO3partArr`` is conceptually an array of :math:`N` ``SO3part`` objects, stored as a single  
    :math:`N\times b\times (2\ell+1)\times n\times 2` dimensional tensor. 

#.  An ``SO3vecArr`` is a collection of such tensors. 



