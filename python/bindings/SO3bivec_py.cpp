// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


py::class_<SO3bivec<float> >(m,"SO3bivec")

    
//.def_static("zero",[](const int b, const SO3type& tau, const int dev){
//  return SO3bivec<float>::zero(b,tau,dev);}, 
//  py::arg("b"), py::arg("tau"), py::arg("device")=0)

//.def_static("gaussian",[](const int b, const SO3type& tau, const int dev){
//  return SO3bivec<float>::gaussian(b,tau,dev);}, 
//  py::arg("b"), py::arg("tau"), py::arg("device")=0)

//  .def(pybind11::init<const vector<at::Tensor>&>())
  .def("torch",[](const SO3bivec<float>& x){return x.torch();})

  .def("add_to_grad",[](SO3bivec<float>& r, const SO3bivec<float>& x){r.add_to_grad(x);})
  .def("get_grad",&SO3bivec<float>::get_grad)

  .def("__len__",[](const SO3bivec<float>& r){return r.size();})
  .def("device",&SO3bivec<float>::device)
  .def("getb",&SO3bivec<float>::getb)
//.def("get_tau",&SO3bivec<float>::get_tau)

  .def("batch",[](SO3bivec<float>& r, int b){return r.batch(b);})
  .def("get_batch_back",[](SO3bivec<float>& r, int b, SO3bivec<float>& x){
      r.get_grad().batch(b).add(x.get_grad());})

  .def("part",[](SO3bivec<float>& r, int l){return SO3bipart<float>(r.part(l));})
  .def("add_to_part_grad_of",[](SO3bivec<float>& r, int l, SO3bipart<float>& x){
      r.get_grad().part(l).add(x.get_grad());})

  .def("add_CGtransform_to",[](SO3bivec<float>& x, const SO3vec<float>& r){
      x.add_CGtransform_to(r);},py::arg("r"))
  .def("add_CGtransform_back",[](SO3bivec<float>& x, SO3vec<float>& g){
      x.get_grad().add_CGtransform_back(g.get_grad());},py::arg("g"))

  .def("str",&SO3bivec<float>::str,py::arg("indent")="")
  .def("__str__",&SO3bivec<float>::str,py::arg("indent")="")
  .def("__repr__",&SO3bivec<float>::repr,py::arg("indent")="")
;

