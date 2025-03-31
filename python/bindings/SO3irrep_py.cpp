/*
 * This file is part of GElib, a C++/CUDA library for group equivariant 
 * tensor operations. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with GElib in the file NONCOMMERICAL.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in orginal
 * or modified form) must retain this copyright notice and must be 
 * accompanied by a verbatim copy of the license. 
 *
 */


py::class_<SO3irrep>(m,"SO3irrep",
  "Class representing irrep of the group SO(3)")

  .def(py::init([](int l){return SO3irrep(l);}))

  .def("matrix",[](const SO3irrep& x, const SO3element<float>& R){return x.matrix(R).torch();})
    
  .def("str",&SO3irrep::str,py::arg("indent")="")
  .def("__str__",&SO3irrep::str,py::arg("indent")="")
  .def("__repr__",&SO3irrep::repr);
