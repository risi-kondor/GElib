
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


py::class_<SO3mvec>(m,"SO3mvec",
  "Class to store b separate SO3 Fourier transforms with k channels")

  .def(pybind11::init<vector<at::Tensor>&>())

  .def_static("raw",[](const int b, const int k, const SO3type& tau, const int dev){
      return SO3mvec::raw(b,k,tau,dev);}, 
    py::arg("b"), py::arg("k"), py::arg("tau"), py::arg("device")=0)

  .def_static("zero",[](const int b, const int k, const SO3type& tau, const int dev){
      return SO3mvec::zero(b,k,tau,dev);}, 
    py::arg("b"), py::arg("k"), py::arg("tau"), py::arg("device")=0)

  .def_static("gaussian",[](const int b, const int k, const SO3type& tau, const int dev){
    return SO3mvec::gaussian(b,tau,dev);}, 
    py::arg("b"), py::arg("k"), py::arg("tau"), py::arg("device")=0)

  .def_static("Fraw",[](const int b, const int k, const int maxl, const int dev){
      return SO3mvec::Fraw(b,k,maxl,dev);}, 
    py::arg("b"), py::arg("k"), py::arg("maxl"), py::arg("device")=0)

  .def_static("Fzero",[](const int b, const int k, const int maxl, const int dev){
      return SO3mvec::Fzero(b,k,maxl,dev);}, 
    py::arg("b"), py::arg("k"), py::arg("maxl"), py::arg("device")=0)

  .def_static("Fgaussian",[](const int b, const int k, const int maxl, const int dev){
      return SO3mvec::Fgaussian(b,k,maxl,dev);}, 
    py::arg("b"), py::arg("k"), py::arg("maxl"), py::arg("device")=0)


  .def_static("view",[](vector<at::Tensor>& v){
      SO3mvec r;
      for(auto& p: v)
	r.parts.push_back(new SO3partB_array(cnine::CtensorB::view(p)));
      return r;
    })

  .def("torch",&SO3mvec::torch)

  .def("add_to_grad",&SO3mvec::add_to_grad)
  .def("add_to_part_of_grad",&SO3mvec::add_to_part_of_grad)
  .def("get_grad",&SO3mvec::get_grad)
  .def("view_of_grad",&SO3mvec::view_of_grad)

  .def("get_dev",&SO3mvec::get_dev)
  .def("getb",&SO3mvec::getb)
  .def("getk",&SO3mvec::getk)
  .def("get_tau",&SO3mvec::get_tau)
  .def("get_maxl",&SO3mvec::get_maxl)
//.def("get_part",&SO3mvec::get_part)

  .def("apply",&SO3mvec::rotate)

  .def("CGproduct",&SO3mvec::CGproduct,py::arg("y"),py::arg("maxl")=-1)
  .def("addCGproduct",[](SO3mvec& r, const SO3mvec& x, const SO3mvec& y){r.add_CGproduct(x,y);},py::arg("x"),py::arg("y"))
  .def("addCGproduct_back0",[](SO3mvec& xg, const SO3mvec& g, const SO3mvec& y){xg.add_CGproduct_back0(g,y);},py::arg("g"),py::arg("y"))
  .def("addCGproduct_back1",[](SO3mvec& yg, const SO3mvec& g, const SO3mvec& x){yg.add_CGproduct_back1(g,x);},py::arg("g"),py::arg("x"))

  .def("DiagCGproduct",&SO3mvec::CGproduct,py::arg("y"),py::arg("maxl")=-1)
  .def("addDiagCGproduct",[](SO3mvec& r, const SO3mvec& x, const SO3mvec& y){r.add_DiagCGproduct(x,y);},py::arg("x"),py::arg("y"))
  .def("addDiagCGproduct_back0",[](SO3mvec& xg, const SO3mvec& g, const SO3mvec& y){xg.add_DiagCGproduct_back0(g,y);},py::arg("g"),py::arg("y"))
  .def("addDiagCGproduct_back1",[](SO3mvec& yg, const SO3mvec& g, const SO3mvec& x){yg.add_DiagCGproduct_back1(g,x);},py::arg("g"),py::arg("x"))

  .def("Fproduct",&SO3mvec::Fproduct,py::arg("y"),py::arg("maxl")=-1)
  .def("addFproduct",[](SO3mvec& r, const SO3mvec& x, const SO3mvec& y){r.add_Fproduct(x,y);},py::arg("x"),py::arg("y"))
  .def("addFproduct_back0",[](SO3mvec& xg, const SO3mvec& g, const SO3mvec& y){xg.add_Fproduct_back0(g,y);},py::arg("g"),py::arg("y"))
  .def("addFproduct_back1",[](SO3mvec& yg, const SO3mvec& g, const SO3mvec& x){yg.add_Fproduct_back1(g,x);},py::arg("g"),py::arg("x"))

//.def("add_iFFT_to",&SO3mvec::add_iFFT_to)
//.def("add_FFT",&SO3mvec::add_FFT)

  .def("device",&SO3mvec::get_device)
  .def("to",&SO3mvec::to_device)
  .def("to_device",&SO3mvec::to_device)
//.def("move_to",[](SO3mvec& x, const int _dev){x.move_to_device(_dev);})

  .def("str",&SO3mvec::str,py::arg("indent")="")
  .def("__str__",&SO3mvec::str,py::arg("indent")="")
  .def("__repr__",&SO3mvec::repr,py::arg("indent")="");

