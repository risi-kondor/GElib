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


py::class_<O3irrep>(m,"O3irrep",
  "Class representing irrep of the group SO(3)")

  .def(py::init([](int l,int p){return O3irrep(O3index(l,p));}))

  .def("matrix",[](const O3irrep& x, float alpha, float beta, float gamma, int p){
	 return x.matrix<float>(alpha,beta,gamma,p).torch();})
  .def("matrix",[](const O3irrep& x, const O3element<float>& R){
	 return x.matrix(R).torch();})
    
  .def("str",&O3irrep::str,py::arg("indent")="")
  .def("__str__",&O3irrep::str,py::arg("indent")="")
  .def("__repr__",&O3irrep::repr);
