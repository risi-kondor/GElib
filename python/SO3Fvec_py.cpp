
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


py::class_<SO3Fvec>(m,"SO3Fvec",
  "Class to store b separate SO3 Fourier transforms")

  .def_static("zero",[](const int b, const int maxl, const int dev){
      return SO3Fvec::zero(b,maxl,dev);},
    py::arg("b"), py::arg("maxl"), py::arg("device")=0)
  .def_static("gaussian",[](const int b, const int maxl, const int dev){
      return SO3Fvec::gaussian(b,maxl,dev);},
    py::arg("b"), py::arg("maxl"), py::arg("device")=0)

//.def("__len__",&SO3Fvec::size)
//.def("type",&SO3Fvec::type)

//  .def("__getitem__",&SO3Fvec::get_part)
//  .def("__setitem__",[](SO3Fvec& obj, const int l, const SO3part& x){
//      obj.set_part(l,x);})

  
  .def(pybind11::init<vector<at::Tensor>&>())
//,[](vector<at::Tensor>& v){
//      SO3Fvec r;
//      for(auto& p: v)
//	r.parts.push_back(new SO3Fpart(cnine::CtensorB::view(p)));
//      return r;
//    })

  .def_static("view",[](vector<at::Tensor>& v){
      SO3Fvec r;
      for(auto& p: v)
	r.parts.push_back(new SO3Fpart(cnine::CtensorB::view(p)));
      return r;
    })

  .def("apply",&SO3Fvec::rotate)

  .def("addFproduct",&SO3Fvec::add_Fproduct,py::arg("x"),py::arg("y"))
  .def("addFproduct_back0",&SO3Fvec::add_Fproduct_back0,py::arg("g"),py::arg("y"))
  .def("addFproduct_back1",&SO3Fvec::add_Fproduct_back1,py::arg("g"),py::arg("x"))

  .def("device",&SO3Fvec::get_device)
  .def("to",&SO3Fvec::to_device)
  .def("to_device",&SO3Fvec::to_device)
//.def("move_to",[](SO3Fvec& x, const int _dev){x.move_to_device(_dev);})

  .def("str",&SO3Fvec::str,py::arg("indent")="")
  .def("__str__",&SO3Fvec::str,py::arg("indent")="")
//.def("__repr__",&SO3Fvec::repr,py::arg("indent")="")
;

