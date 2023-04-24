d// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


py::class_<SO3vecArray<float> >(m,"SO3vecArray")

    
  .def_static("zero",[](const int b, const SO3type& tau, const int dev){
    return SO3vecArray<float>::zero(b,tau,dev);}, 
    py::arg("b"), py::arg("tau"), py::arg("device")=0)

  .def_static("gaussian",[](const int b, const SO3type& tau, const int dev){
    return SO3vecArray<float>::gaussian(b,tau,dev);}, 
    py::arg("b"), py::arg("tau"), py::arg("device")=0)

  .def(pybind11::init<const vector<at::Tensor>&>())
  .def("torch",[](const SO3vecArray<float>& x){return x.torch();})

  .def("add_to_grad",[](SO3vecArray<float>& r, const SO3vecArray<float>& x){r.add_to_grad(x);})
  .def("get_grad",&SO3vecArray<float>::get_grad)

  .def("__len__",[](const SO3vecArray<float>& r){return r.size();})
  .def("device",&SO3vecArray<float>::device)
  .def("getb",&SO3vecArray<float>::getb)
  .def("get_tau",&SO3vecArray<float>::get_tau)

  .def("add_CGproduct",[](SO3vecArray<float>& r, const SO3vecArray<float>& x, const SO3vecArray<float>& y){
      r.add_CGproduct(x,y);},py::arg("x"),py::arg("y"))
  .def("add_CGproduct_back0",[](SO3vecArray<float>& r, SO3vecArray<float>& g, const SO3vecArray<float>& y){
      r.get_grad().add_CGproduct_back0(g.get_grad(),y);},py::arg("g"),py::arg("y"))
  .def("add_CGproduct_back1",[](SO3vecArray<float>& r, SO3vecArray<float>& g, const SO3vecArray<float>& x){
      r.get_grad().add_CGproduct_back1(g.get_grad(),x);},py::arg("g"),py::arg("x"))

  .def("str",&SO3vecArray<float>::str,py::arg("indent")="")
  .def("__str__",&SO3vecArray<float>::str,py::arg("indent")="")
  .def("__repr__",&SO3vecArray<float>::repr,py::arg("indent")="")
;

