
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


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

  
