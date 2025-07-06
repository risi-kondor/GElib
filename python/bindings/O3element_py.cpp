/*
 * This file is part of GElib, a C++/CUDA library for group equivariant 
 * tensor operations. 
 *  
 * Copyright (c) 2025, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with GElib in the file NONCOMMERICAL.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in orginal
 * or modified form) must retain this copyright notice and must be 
 * accompanied by a verbatim copy of the license. 
 *
 */

py::class_<O3element<float> >(m,"O3element",
  "Class to store an elelement of the group O(3)")

  .def_static("identity",[](){return O3element<float>::identity();})
  .def_static("random",[](){return O3element<float>::random();})

  .def_static("view",[](at::Tensor& x){
      return O3element<float>(tensorf::view(x));})

  .def("torch",&O3element<float>::torch)
  .def("parity",&O3element<float>::parity)

  .def("str",&O3element<float>::str,py::arg("indent")="")
  .def("__str__",&O3element<float>::str,py::arg("indent")="")
  .def("__repr__",&O3element<float>::repr);

  
