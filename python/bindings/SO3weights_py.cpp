
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


py::class_<SO3weights>(m,"SO3weights",
  "Vector of weight matrices")

  .def_static("raw",[](const vector<int>& tau1, const vector<int>& tau2, const int dev){
      return SO3weights::raw(tau1,tau2,dev);},py::arg("tau1"),py::arg("tau2"),py::arg("device")=0)

  
// ---- Conversions, transport, etc. ------------------------------------------------------------------------


  .def(pybind11::init<vector<at::Tensor>&>())
  .def_static("view",[](vector<at::Tensor>& v){return CtensorPackObj::view(v);})
  .def("torch",&CtensorPackObj::torch)

  .def("add_to_grad",&CtensorPackObj::add_to_grad)
  .def("get_grad",&CtensorPackObj::get_grad)
  .def("view_of_grad",&CtensorPackObj::view_of_grad)

  .def("device",&CtensorPackObj::get_dev)
  .def("to",&CtensorPackObj::to_device)
  .def("to_device",&CtensorPackObj::to_device)
  .def("move_to",[](CtensorPackObj& x, const int _dev){x.move_to_device(_dev);})


// ---- Cumulative operations --------------------------------------------------------------------------------


  .def("add",[](CtensorPackObj& x, const CtensorPackObj& y, const complex<float> c){x.add(y,c);})
  .def("add_conj",[](CtensorPackObj& x, const CtensorPackObj& y, const complex<float> c){x.add(y,c);})


// ---- I/O --------------------------------------------------------------------------------------------------


  .def("str",&CtensorPackObj::str,py::arg("indent")="")
  .def("__str__",&CtensorPackObj::str,py::arg("indent")="")
  .def("__repr__",&CtensorPackObj::str,py::arg("indent")="")

;


