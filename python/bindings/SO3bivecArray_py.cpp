// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


py::class_<SO3bivecArray<float> >(m,"SO3bivecArray")

    
  .def_static("zero",[](const int b, const vector<int>& adims, const SO3bitype& tau, const int dev){
      return SO3bivecArray<float>::zero(b,adims,tau,dev);}, 
    py::arg("b"), py::arg("adims"), py::arg("tau"), py::arg("device")=0)

  .def_static("gaussian",[](const int b, const vector<int>& adims, const SO3bitype& tau, const int dev){
      return SO3bivecArray<float>::gaussian(b,adims,tau,dev);}, 
    py::arg("b"), py::arg("adims"), py::arg("tau"), py::arg("device")=0)

//  .def(pybind11::init<const vector<at::Tensor>&>())
  .def("torch",[](const SO3bivecArray<float>& x){return x.torch();})

  .def("add_to_grad",[](SO3bivecArray<float>& r, const SO3bivecArray<float>& x){r.add_to_grad(x);})
  .def("get_grad",[] (SO3bivecArray<float>& arr) { return arr.get_grad(); })

  .def("__len__",[](const SO3bivecArray<float>& r){return r.size();})
  .def("device",&SO3bivecArray<float>::device)
  .def("getb",&SO3bivecArray<float>::getb)
  .def("get_adims",&SO3bivecArray<float>::get_adims)
//  .def("get_tau",&SO3bivecArray<float>::get_tau)

  .def("batch",[](SO3bivecArray<float>& r, int b){return r.batch(b);})
  .def("get_batch_back",[](SO3bivecArray<float>& r, int b, SO3bivecArray<float>& x){
      r.get_grad().batch(b).add(x.get_grad());})

  .def("part",[](SO3bivecArray<float>& r, int l){return SO3bipartArray<float>(r.part(l));})
  .def("get_part_back",[](SO3bivecArray<float>& r, int l, SO3bipartArray<float>& x){
      r.get_grad().part(l).add(x.get_grad());})

  .def("cell",[](SO3bivecArray<float>& r, vector<int>& ix){return SO3bivec<float>(r.cell(ix));})
  .def("get_cell_back",[](SO3bivecArray<float>& r, vector<int>& ix, SO3bivec<float>& x){
      r.get_grad().cell(ix).add(x.get_grad());})

  .def("add_CGtransform_to",[](SO3bivecArray<float>& x, const SO3vecArray<float>& r){
      x.add_CGtransform_to(r);},py::arg("r"))
  .def("add_CGproduct_back",[](SO3bivecArray<float>& x, SO3vecArray<float>& g){
      x.get_grad().add_CGtransform_back(g.get_grad());},py::arg("g"))

  .def("str",&SO3bivecArray<float>::str,py::arg("indent")="")
  .def("__str__",&SO3bivecArray<float>::str,py::arg("indent")="")
  .def("__repr__",&SO3bivecArray<float>::repr,py::arg("indent")="")
;

