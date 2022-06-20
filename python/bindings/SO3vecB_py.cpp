
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


py::class_<SO3vecB>(m,"SO3vecB",
  "Class to store b separate SO3 Fourier transforms")

  .def_static("raw",static_cast<SO3vecB(*)(const int, const SO3type&, const int)>(&SO3vecB::raw))
  .def_static("zero",static_cast<SO3vecB(*)(const int, const SO3type&, const int)>(&SO3vecB::zero))
  .def_static("gaussian",static_cast<SO3vecB(*)(const int, const SO3type&, const int)>(&SO3vecB::gaussian))

  .def_static("Fraw",[](const int b, const int maxl, const int dev){
    return SO3vecB::Fraw(b,maxl,dev);}, 
    py::arg("b"), py::arg("maxl"), py::arg("device")=0)

//.def_static("Fraw",static_cast<SO3vecB(*)(const int, const int, const int)>(&SO3vecB::Fraw))
//  .def_static("Fzero",static_cast<SO3vecB(*)(const int, const int)>(&SO3vecB::Fzero))
//  .def_static("Fgaussian",static_cast<SO3vecB(*)(const int, const int)>(&SO3vecB::Fgaussian))

  .def(pybind11::init<vector<at::Tensor>&>())

  .def_static("view",[](vector<at::Tensor>& v){
      SO3vecB r;
      for(auto& p: v)
	r.parts.push_back(static_cast<SO3partB*>(cnine::CtensorB::viewp(p)));
      return r;
    })

  .def("get_dev",&SO3vecB::get_dev)
  .def("getb",&SO3vecB::getb)
  .def("get_tau",&SO3vecB::get_tau)
  .def("get_maxl",&SO3vecB::get_maxl)

  .def("apply",&SO3vecB::rotate)

  .def("addCGproduct",&SO3vecB::add_CGproduct,py::arg("x"),py::arg("y"))
  .def("addCGproduct_back0",&SO3vecB::add_CGproduct_back0,py::arg("g"),py::arg("y"))
  .def("addCGproduct_back1",&SO3vecB::add_CGproduct_back1,py::arg("g"),py::arg("x"))

  .def("addDiagCGproduct",&SO3vecB::add_DiagCGproduct,py::arg("x"),py::arg("y"))
  .def("addDiagCGproduct_back0",&SO3vecB::add_DiagCGproduct_back0,py::arg("g"),py::arg("y"))
  .def("addDiagCGproduct_back1",&SO3vecB::add_DiagCGproduct_back1,py::arg("g"),py::arg("x"))

  .def("addFproduct",&SO3vecB::add_Fproduct,py::arg("x"),py::arg("y"),py::arg("method")=0)
  .def("addFproduct_back0",&SO3vecB::add_Fproduct_back0,py::arg("g"),py::arg("y"),py::arg("method")=0)
  .def("addFproduct_back1",&SO3vecB::add_Fproduct_back1,py::arg("g"),py::arg("x"))

  .def("add_iFFT_to",&SO3vecB::add_iFFT_to)
  .def("add_FFT",&SO3vecB::add_FFT)

  .def("device",&SO3vecB::get_device)
  .def("to",&SO3vecB::to_device)
  .def("to_device",&SO3vecB::to_device)
//.def("move_to",[](SO3vecB& x, const int _dev){x.move_to_device(_dev);})

  .def("str",&SO3vecB::str,py::arg("indent")="")
  .def("__str__",&SO3vecB::str,py::arg("indent")="")
  .def("__repr__",&SO3vecB::repr,py::arg("indent")="");

