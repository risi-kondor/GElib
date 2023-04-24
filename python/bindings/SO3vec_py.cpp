// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


py::class_<SO3vec<float> >(m,"SO3vec")

    
  .def_static("zero",[](const int b, const SO3type& tau, const int dev){
    return SO3vec<float>::zero(b,tau,dev);}, 
    py::arg("b"), py::arg("tau"), py::arg("device")=0)

  .def_static("gaussian",[](const int b, const SO3type& tau, const int dev){
    return SO3vec<float>::gaussian(b,tau,dev);}, 
    py::arg("b"), py::arg("tau"), py::arg("device")=0)

  .def(pybind11::init<const vector<at::Tensor>&>())
  .def("torch",[](const SO3vec<float>& x){return x.torch();})

  .def("add_to_grad",[](SO3vec<float>& r, const SO3vec<float>& x){r.add_to_grad(x);})
  .def("get_grad",&SO3vec<float>::get_grad)

  .def("__len__",[](const SO3vec<float>& r){return r.size();})
  .def("device",&SO3vec<float>::device)
  .def("getb",&SO3vec<float>::getb)
  .def("get_tau",&SO3vec<float>::get_tau)

  .def("add_CGproduct",[](SO3vec<float>& r, const SO3vec<float>& x, const SO3vec<float>& y){
      r.add_CGproduct(x,y);},py::arg("x"),py::arg("y"))
  .def("add_CGproduct_back0",[](SO3vec<float>& r, SO3vec<float>& g, const SO3vec<float>& y){
      r.get_grad().add_CGproduct_back0(g.get_grad(),y);},py::arg("g"),py::arg("y"))
  .def("add_CGproduct_back1",[](SO3vec<float>& r, SO3vec<float>& g, const SO3vec<float>& x){
      r.get_grad().add_CGproduct_back1(g.get_grad(),x);},py::arg("g"),py::arg("x"))

  .def("str",&SO3vec<float>::str,py::arg("indent")="")
  .def("__str__",&SO3vec<float>::str,py::arg("indent")="")
  .def("__repr__",&SO3vec<float>::repr,py::arg("indent")="")
;

