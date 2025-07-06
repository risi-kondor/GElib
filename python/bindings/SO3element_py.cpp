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

py::class_<SO3element<float> >(m,"SO3element",
  "Class to store an elelement of the group SO(3)")

  .def_static("identity",[](){return SO3element<float>::identity();})
  .def_static("random",[](){return SO3element<float>::random();})

  .def_static("view",[](at::Tensor& x){
      return SO3element<float>(tensorf::view(x));})

  .def("torch",&SO3element<float>::torch)

  .def("str",&SO3element<float>::str,py::arg("indent")="")
  .def("__str__",&SO3element<float>::str,py::arg("indent")="")
  .def("__repr__",&SO3element<float>::repr);

  
