
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


py::class_<SO3bipartArray<float> >(m,"SO3bipartArray",
  "Class to store an array consisting of n vectors transforming according to a specific irreducible representation of SO(3)")

    
  .def_static("raw",[](const int b, const vector<int>& adims, const int l1, const int l2, const int n, const int dev){
      return SO3bipartArray<float>::raw(b,adims,l1,l2,n,dev);}, 
    py::arg("b"), py::arg("adims"), py::arg("l1"), py::arg("l2"), py::arg("n")=1, py::arg("device")=0)
  .def_static("zero",[](const int b, const vector<int>& adims, const int l1, const int l2, const int n, const int dev){
      return SO3bipartArray<float>::zero(b,adims,l1,l2,n,dev);}, 
    py::arg("b"), py::arg("adims"), py::arg("l1"), py::arg("l2"), py::arg("n")=1, py::arg("device")=0)
  .def_static("gaussian",[](const int b, const vector<int>& adims, const int l1, const int l2, const int n, const int dev){
      return SO3bipartArray<float>::gaussian(b,adims,l1,l2,n,dev);}, 
    py::arg("b"), py::arg("adims"), py::arg("l1"), py::arg("l2"), py::arg("n")=1, py::arg("device")=0)

//.def(pybind11::init([](const at::Tensor& x){return SO3bipartArray<float>(cnine::CtensorB(x));}))
  .def(pybind11::init<const at::Tensor&>())
  .def("torch",[](const SO3bipartArray<float>& x){return x.torch();})

  .def("add_to_grad",[](SO3bipartArray<float>& r, const SO3bipartArray<float>& x){r.add_to_grad(x);})
  .def("get_grad",[] (SO3bipartArray<float>& arr) {return arr.get_grad(); })

  .def("__len__",[](const SO3bipartArray<float>& obj){return 1;})
  .def("device",&SO3bipartArray<float>::device)
  .def("getb",&SO3bipartArray<float>::getb)
  //.def("get_adims",[](const SO3bipartArray<float>& x){return vector<int>(x.get_adims());})
  .def("get_adims",&SO3bipartArray<float>::get_adims) 
  .def("getl1",&SO3bipartArray<float>::getl1)
  .def("getl2",&SO3bipartArray<float>::getl2)
  .def("getn",&SO3bipartArray<float>::getn)

  .def("batch",[](SO3bipartArray<float>& r, int b){return r.batch(b);})
  .def("get_batch_back",[](SO3bipartArray<float>& r, int b, SO3bipartArray<float>& x){
      r.get_grad().batch(b).add(x.get_grad());})

  .def("cell",[](SO3bipartArray<float>& r, vector<int>& ix){return SO3bipart<float>(r.cell(ix));})
  .def("get_cell_back",[](SO3bipartArray<float>& r, vector<int>& ix, SO3bipart<float>& x){
      r.get_grad()(ix).add(x.get_grad());})

  .def("add_CGtransform_to",[](SO3bipartArray<float>& x, SO3partArray<float>& r, const int offs){
      x.add_CGtransform_to(r,offs);},py::arg("r"),py::arg("offs")=0)
  .def("add_CGtransform_back",[](SO3bipartArray<float>& x, SO3partArray<float>& g, const int offs){
      x.get_grad().add_CGtransform_back(g.get_grad(),offs);},py::arg("g"),py::arg("offs")=0)
    
  .def("str",&SO3bipartArray<float>::str,py::arg("indent")="")
  .def("__str__",&SO3bipartArray<float>::str,py::arg("indent")="")
  .def("__repr__",&SO3bipartArray<float>::repr,py::arg("indent")="")
;


// ---- Stand-alone functions --------------------------------------------------------------------------------

    
m.def("CGtransform",[](const SO3bipartArray<float>& x, const int l){
    return CGtransform(x,l);});


