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

py::class_<cnine::GatherMapB>(m,"gather_map","Class to store the type of an SO3-vector")

  .def_static("sources_targets",[](const vector<int> sources, const vector<int> targets){
      return cnine::GatherMapB(sources,targets);})

  .def_static("from_matrix",[](const at::Tensor& M, const int nin, const int nout){
      return cnine::GatherMapB(TensorView<int>::view(M),nin,nout);})

  .def_static("random",[](const int n, const int m, const float p){
      return cnine::GatherMapB::random(n,m,p);},py::arg("n"),py::arg("m"),py::arg("p")=0.5)

  .def("str",&cnine::GatherMapB::str,py::arg("indent")="")
  .def("__str__",&cnine::GatherMapB::str,py::arg("indent")="")
  .def("__repr__",&cnine::GatherMapB::repr)
;
