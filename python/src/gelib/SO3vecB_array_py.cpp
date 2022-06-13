
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


py::class_<SO3vecB_array>(m,"SO3vecB_array",
  "Class to store an array of SO3-vectors")

  .def_static("zero",[](const Gdims& adims, const SO3type& tau, const int dev){
      return SO3vecB_array::zero(adims,tau,dev);}, 
    py::arg("adims"), py::arg("tau"), py::arg("device")=0)
  .def_static("zero",[](const vector<int>& av, const SO3type& tau, const int dev){
      return SO3vecB_array::zero(Gdims(av),tau,dev);},
    py::arg("adims"),py::arg("tau"),py::arg("device")=0)

  .def_static("view",[](vector<at::Tensor>& v){
      SO3vecB_array r;
      for(auto& p: v)
	r.parts.push_back(static_cast<SO3partB_array*>(SO3partB_array::viewp(p,-2)));
      //r.parts.push_back(static_cast<SO3partB_array*>(SO3partB_array::viewp(p)));
      return r;
    })

  .def("apply",&SO3vecB_array::rotate)

  .def("addCGproduct",&SO3vecB_array::add_CGproduct,py::arg("x"),py::arg("y"))
  .def("addCGproduct_back0",&SO3vecB_array::add_CGproduct_back0,py::arg("g"),py::arg("y"))
  .def("addCGproduct_back1",&SO3vecB_array::add_CGproduct_back1,py::arg("g"),py::arg("x"))

//.def("gather",&SO3vecB_array::add_gather,py::arg("x"),py::arg("mask"))
  .def("gather",[](SO3vecB_array& x, const SO3vecB_array& y, const cnine::Rmask1& mask){
      x.add_gather(y,mask);},py::arg("x"),py::arg("mask"))

  .def("device",&SO3vecB_array::get_device)
  .def("to",&SO3vecB_array::to_device)
  .def("to_device",&SO3vecB_array::to_device)
//.def("move_to",[](SO3vecB_array& x, const int _dev){x.move_to_device(_dev);})

  .def("str",&SO3vecB_array::str,py::arg("indent")="")
  .def("__str__",&SO3vecB_array::str,py::arg("indent")="")
  .def("__repr__",&SO3vecB_array::repr,py::arg("indent")="");

