
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

  .def_static("zero",[](const int b, const Gdims& adims, const SO3type& tau, const int dev){
      return SO3vecB_array::zero(b,adims,tau,dev);}, 
    py::arg("b"),py::arg("adims"), py::arg("tau"), py::arg("device")=0)
  .def_static("zero",[](const int b, const vector<int>& av, const SO3type& tau, const int dev){
      return SO3vecB_array::zero(b,Gdims(av),tau,dev);},
    py::arg("b"),py::arg("adims"),py::arg("tau"),py::arg("device")=0)

  .def_static("view",[](vector<at::Tensor>& v){
      SO3vecB_array r;
      for(auto& p: v)
	r.parts.push_back(static_cast<SO3partB_array*>(SO3partB_array::viewp(p,-2,true)));
      return r;
    })

  .def("apply",&SO3vecB_array::rotate)

  .def("addCGproduct",&SO3vecB_array::add_CGproduct,py::arg("x"),py::arg("y"))
  .def("addCGproduct_back0",&SO3vecB_array::add_CGproduct_back0,py::arg("g"),py::arg("y"))
  .def("addCGproduct_back1",&SO3vecB_array::add_CGproduct_back1,py::arg("g"),py::arg("x"))

  .def("addDiagCGproduct",&SO3vecB_array::add_DiagCGproduct,py::arg("x"),py::arg("y"))
  .def("addDiagCGproduct_back0",&SO3vecB_array::add_DiagCGproduct_back0,py::arg("g"),py::arg("y"))
  .def("addDiagCGproduct_back1",&SO3vecB_array::add_DiagCGproduct_back1,py::arg("g"),py::arg("x"))

  .def("addDiagCGproductB",&SO3vecB_array::add_DiagCGproductB,py::arg("x"),py::arg("y"))
  .def("addDiagCGproductB_back0",&SO3vecB_array::add_DiagCGproductB_back0,py::arg("g"),py::arg("y"))
  .def("addDiagCGproductB_back1",&SO3vecB_array::add_DiagCGproductB_back1,py::arg("g"),py::arg("x"))

  .def("addDDiagCGproduct",&SO3vecB_array::add_DDiagCGproduct,py::arg("x"),py::arg("y"))
  .def("addDDiagCGproduct_back0",&SO3vecB_array::add_DDiagCGproduct_back0,py::arg("g"),py::arg("y"))
  .def("addDDiagCGproduct_back1",&SO3vecB_array::add_DDiagCGproduct_back1,py::arg("g"),py::arg("x"))

  .def("Fproduct",&SO3vecB_array::Fproduct,py::arg("y"),py::arg("maxl")=-1)
  .def("addFproduct",&SO3vecB_array::add_Fproduct,py::arg("x"),py::arg("y"),py::arg("method")=0)
  .def("addFproduct_back0",&SO3vecB_array::add_Fproduct_back0,py::arg("g"),py::arg("y"),py::arg("method")=0)
  .def("addFproduct_back1",&SO3vecB_array::add_Fproduct_back1,py::arg("g"),py::arg("x"))

//.def("gather",&SO3vecB_array::add_gather,py::arg("x"),py::arg("mask"))
  .def("add_gather",[](SO3vecB_array& x, const SO3vecB_array& y, const cnine::Rmask1& mask){
      x.add_gather(y,mask);},py::arg("x"),py::arg("mask"))

  .def("device",&SO3vecB_array::get_device)
  .def("to",&SO3vecB_array::to_device)
  .def("to_device",&SO3vecB_array::to_device)
//.def("move_to",[](SO3vecB_array& x, const int _dev){x.move_to_device(_dev);})

  .def("str",&SO3vecB_array::str,py::arg("indent")="")
  .def("__str__",&SO3vecB_array::str,py::arg("indent")="")
  .def("__repr__",&SO3vecB_array::repr,py::arg("indent")="");

