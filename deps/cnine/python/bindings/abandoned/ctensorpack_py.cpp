/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2021, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


pybind11::class_<CtensorPackObj>(m,"ctensorpack")

  .def_static("raw",[](const int n, const Gdims& dims, const int dev){return CtensorPackObj::raw(n,dims,dev);},
    py::arg("n"),py::arg("dims"),py::arg("device")=0)
  .def_static("zero",[](const int n, const Gdims& dims, const int dev){return CtensorPackObj::zero(n,dims,dev);},
    py::arg("n"),py::arg("dims"),py::arg("device")=0)
  .def_static("ones",[](const int n, const Gdims& dims, const int dev){return CtensorPackObj::ones(n,dims,dev);},
    py::arg("n"),py::arg("dims"),py::arg("device")=0)
  .def_static("gaussian",[](const int n, const Gdims& dims, const int dev){return CtensorPackObj::gaussian(n,dims,dev);},
    py::arg("n"),py::arg("dims"),py::arg("device")=0)
  .def_static("sequential",[](const int n, const Gdims& dims, const int dev){return CtensorPackObj::sequential(n,dims,dev);},
    py::arg("n"),py::arg("dims"),py::arg("device")=0)


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
  .def("__repr__",&CtensorPackObj::str,py::arg("indent")="");





