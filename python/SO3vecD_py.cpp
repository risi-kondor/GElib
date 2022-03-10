
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


py::class_<SO3vecD>(m,"SO3vecD",
  "Class to store b separate SO3 Fourier transforms")

  .def_static("zero",static_cast<SO3vecD(*)(const int, const int, const SO3type&, const int)>(&SO3vecD::zero))
//.def_static("ones",static_cast<SO3vecD(*)(const int, const SO3type&, const int)>(&SO3vecD::ones))
  .def_static("gaussian",static_cast<SO3vecD(*)(const int, const int, const SO3type&, const int)>(&SO3vecD::gaussian))

//.def("__len__",&SO3vecD::size)
//.def("type",&SO3vecD::type)

//  .def("__getitem__",&SO3vecD::get_part)
//  .def("__setitem__",[](SO3vecD& obj, const int l, const SO3part& x){
//      obj.set_part(l,x);})

  
  .def(pybind11::init<vector<at::Tensor>&>())
//,[](vector<at::Tensor>& v){
//      SO3vecD r;
//      for(auto& p: v)
//	r.parts.push_back(new SO3Fpart(cnine::CtensorB::view(p)));
//      return r;
//    })

  .def_static("view",[](vector<at::Tensor>& v){
      SO3vecD r;
      for(auto& p: v)
	r.parts.push_back(static_cast<SO3partD*>(cnine::CtensorB::viewp(p)));
      //r.parts.push_back(new SO3partD(cnine::CtensorB::view(p)));
      return r;
    })

  .def("apply",&SO3vecD::rotate)

  .def("addCGproduct",&SO3vecD::add_CGproduct,py::arg("x"),py::arg("y"))
  .def("addCGproduct_back0",&SO3vecD::add_CGproduct_back0,py::arg("g"),py::arg("y"))
  .def("addCGproduct_back1",&SO3vecD::add_CGproduct_back1,py::arg("g"),py::arg("x"))

  .def("gather",&SO3vecD::add_gather,py::arg("x"),py::arg("mask"))

  .def("device",&SO3vecD::get_device)
  .def("to",&SO3vecD::to_device)
  .def("to_device",&SO3vecD::to_device)
//.def("move_to",[](SO3vecD& x, const int _dev){x.move_to_device(_dev);})

  .def("str",&SO3vecD::str,py::arg("indent")="")
  .def("__str__",&SO3vecD::str,py::arg("indent")="")
  .def("__repr__",&SO3vecD::repr,py::arg("indent")="");

