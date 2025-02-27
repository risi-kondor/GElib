***********************
Mathematical background
***********************


A finite dimensional **representation** of a group :math:`G` over a field :math:`\mathbb{F}` is a 
matrix valued function :math:`\rho\colon G\to\mathbb{F}^{d\times d}` such that 

.. math::
  \rho(xy)=\rho(x)\,\rho(y)

for any :math:`x,y\in G`. While it is possible to define representations over different fields, 
in `GElib` we only consider complex valued representations (:math:`\mathbb{F}=\mathbb{C}`) 
which is the most common and generally the most straightforward choice. 
 
Different representations of the same group can be related to each other in various ways, for example, 
:math:`\rho` might decompose into a direct sum of smaller representations in the form 

.. math::
  \rho(x)=\left(\begin{array}{c|c} \rho_1(x)& \\ \hline &\rho_2(x)\end{array}\right) 

However, provided that :math:`G` is a 'compact` group, it always has a finite or countable sequence of 
elementary representations :math:`\rho_1,\rho_2,\ldots` called its **irreducible representations** 
(or just **irreps** for short) such that any representation of :math:`G` can be reduced into a direct 
sum of some subset of :math:`\rho_1,\rho_2,\ldots`. 


===========================
Equivariant neural networks 
===========================

Equivariant neural networks are a class of neural architectures that are designed to strictly obey certain 
underlying symmetries of the learning problem, such as invariance to translations, rotations, and so on. 
In order to achieve this, equivariant archtectures store information in objects that transform 
according to the individual irreps of the group :math:`G` corresponding to the underlying symmetry. 
Technically, such objects are called **G-covariant**. 

.. For a given group :math:`G`, 

`GElib` provides two basic classes to facilitate working G-covariant objects:

#. The ``Gpart`` class stores a collection of :math:`n` vectors transforming according to the `same` 
   irrep :math:`\rho` of :math:`G`. Denoting the dimensionality of :math:`\rho` by :math:`d_\rho`, a ``Gpart`` object can be 
   thought of as a :math:`d_\rho\times n` dimensional matrix :math:`F`, and the group acts on :math:`F` by 
   :math:`F\mapsto \rho(x) F`. 

#. The ``Gvec`` object stores a collection of ``Gpart``\s corresponding to different irreps. Hence it can 
   store more general :math:`G`-covariant objects that transform according to any representation. 

Here ``G`` is to be replaced by the name of the actual group. For example, in the case of the three dimensional 
rotation group ``SO3`` the relevant `GElib` classes are ``SO3part`` and ``SO3vec``. 
Since the irreps are always assumed to be complex valued, the ``Gpart`` and ``Gvec`` objects are also complex. 

`GElib` provides various facilities for operating on G-covariant objects. 
The most important of these is the *Clebsch--Gordan product* described below. 

===========================
The Clebsch--Gordan product
===========================

Given any two irreducible representations :math:`\rho_a` and :math:`\rho_b` of :math:`G`, their tensor  
(Kronecker) product :math:`(\rho_a\!\otimes\!\rho_b)(x):=\rho_a(x)\!\otimes\!\rho_b(x)` is also a representation, but,  
in general, it is not irreducible. By the general theory of representations of compact groups it must 
then be reducible to a direct sum of irreps in the form 

.. math::
   :label: CGdecomp

   \rho_a(x)\otimes\rho_b(x)=C^\dag_{a,b} \bigg[\bigoplus_i \bigoplus_{j=1}^{\kappa(a,b,i)} \rho_i(x) \bigg] C_{a,b}

for some combination of :math:`\kappa(a,b,i)` multiplicities and :math:`C_{a,b}` unitary transformation matrix. 
This is called the **Clebsch-Gordan decomposition**. 

What irreps exactly appear in the decomposition, what the multiplicities are, and what the matrix :math:`C_{a,b}` 
is are nontrivial questions that must be derived on a group-by-group basis. For the groups most commonly 
used in equivariant nets however the answer to these questions is can generally be readily found in the mathematics literature. 
For the purposes of software like GElib the decomposition can be hard coded, since it depends on the group, 
and not any form of "data". 


The significance of the Clebsch--Gordan transform for equivariant nets is that it also prescribes 
the way that the tensor product of G-covariant 
parts and vectors transform. In particular, if :math:`F_a` and :math:`F_b` are two G-covariant parts corresponding 
to :math:`\rho_a` resp. :math:`\rho_b`, i.e., they transform under the action of the group as 

.. math::
   F_a\longmapsto \rho_a(g)\,F_a \hspace{50pt} F_b\longmapsto \rho_b(g)\,F_b, 

then their tensor product transforms as 

.. math::
  F_a\otimes F_b \longmapsto (\rho_a(x)\otimes\rho_b(x))(F_a\otimes F_b).  

However, if we break :math:`C_{a,b}` into a sequence of horizontal blocks :math:`C_{a,b}^{i,j}` 
corresponding to the block structure of the matrix appearing in the square brackets in 
:eq:`CGdecomp`, and form the so-called **Clebsch--Gordan product** (**CG-product** for short) matrices 

.. math::
   F_a\otimes_{i,j} F_b:=C_{a,b}^{i,j}(F_a\otimes F_b), 

it is relatively easy to see that each of these matrices transforms simply as 

.. math::
   F_a\otimes_{i,j}\! F_b\longmapsto \rho_i(x) (F_a\otimes_{i,j}\! F_b), 

i.e., each :math:`F_a\otimes_{i,j}\! F_b` is itself an :math:`\rho_i`--covariant part. 
Thus, the Clebsch--Gordan product is a bilinear operation on G-covariant objects that produces 
other G-covariant objects. 

The concept of Clebsch--Gordan product also readily extends to ``Gvec``\s. To compute the CG-product 
of two ``Gvec``\s :math:`\mathbf{u}` and :math:`\mathbf{v}`, we simply compute the CG-product of each 
part in :math:`\mathbf{u}` with each part in :math:`\mathbf{v}` and concatenate all components 
that correspond to the same irrep. 

Due to their ability to nontrivially combine objects transforming according to the different irreps of 
the same group Clebsch--Gordan products play a ciritical role in equivariant neural architectures. 
Unfortunately, they are also relatively costly, and often end up becoming the computational bottleneck. 
One of the most important components of `GElib` is providing fast implementations of CG-products. 
 
 

   


