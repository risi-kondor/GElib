
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


py::class_<SO3bipart<float> >(m,"SO3bipart",
  "Class to store an array consisting of n vectors transforming according to a specific irreducible representation of SO(3)")

  .def(pybind11::init<const at::Tensor&>())
    
  .def_static("raw",[](const int b, const int l1, const int l2, const int n, const int dev){
      return SO3bipart<float>::raw(b,l1,l2,n,dev);}, py::arg("b"), py::arg("l1"), py::arg("l2"), py::arg("n")=1, py::arg("device")=0)
  .def_static("zero",[](const int b, const int l1, const int l2, const int n, const int dev){
      return SO3bipart<float>::zero(b,l1,l2,n,dev);}, py::arg("b"), py::arg("l1"), py::arg("l2"),py::arg("n")=1, py::arg("device")=0)
  .def_static("gaussian",[](const int b, const int l1, const int l2, const int n, const int dev){
      return SO3bipart<float>::gaussian(b,l1,l2,n,dev);}, py::arg("b"), py::arg("l1"), py::arg("l2"),py::arg("n")=1, py::arg("device")=0)

//.def(pybind11::init([](const at::Tensor& x){return SO3part<float>(cnine::CtensorB(x));}))
//  .def_static("view",[](at::Tensor& x){return SO3part<float>(cnine::CtensorB::view(x));})
  .def("torch",[](const SO3bipart<float>& x){return x.torch();})

  .def("add_to_grad",[](SO3bipart<float>& r, const SO3bipart<float>& x){r.add_to_grad(x);})
  .def("get_grad",&SO3bipart<float>::get_grad)

  .def("__len__",[](const SO3bipart<float>& obj){return 1;})
  .def("device",&SO3bipart<float>::device)
  .def("getb",&SO3bipart<float>::getb)
  .def("getl",&SO3bipart<float>::getl1)
  .def("getl",&SO3bipart<float>::getl2)
  .def("getn",&SO3bipart<float>::getn)

  .def("batch",[](SO3bipart<float>& r, int b){return r.batch(b);})
  .def("get_batch_back",[](SO3bipart<float>& r, int b, SO3bipart<float>& x){
      r.get_grad().batch(b).add(x.get_grad());})

  .def("add_CGtranform_to",[](SO3bipartView<float>& x, SO3partView<float>& r, int offs){
      x.add_CGtransform_to(r,offs);},py::arg("r"),py::arg("offs")=0)
  .def("add_CGtransform_back",[](SO3bipart<float>& x, SO3part<float>& r, const int offs){
      x.get_grad().add_CGtransform_back(r.get_grad(),offs);},py::arg("g"),py::arg("offs")=0)
    
  .def("str",&SO3bipart<float>::str,py::arg("indent")="")
  .def("__str__",&SO3bipart<float>::str,py::arg("indent")="")
  .def("__repr__",&SO3bipart<float>::repr,py::arg("indent")="")
;


// ---- Stand-alone functions --------------------------------------------------------------------------------

    
m.def("CGtransform",[](const SO3bipart<float>& x, const int l){
    return CGtransform(x,l);});


