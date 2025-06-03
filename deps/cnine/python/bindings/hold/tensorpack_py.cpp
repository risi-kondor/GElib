/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


pybind11::class_<TensorPack<float> >(m,"tensorpack")

  .def_static("zero",[](const int n, const vector<int>& v, const int dev){
      return TensorPack<float>::zero(n,Gdims(v),dev);},
    py::arg("n"),py::arg("dims"),py::arg("device")=0)

  .def_static("zeros",[](const int n, const vector<int>& v, const int dev){
      return TensorPack<float>::zero(n,Gdims(v),dev);},
    py::arg("n"),py::arg("dims"),py::arg("device")=0)

  .def_static("sequential",[](const int n, const vector<int>& v, const int dev){
      return TensorPack<float>::sequential(n,Gdims(v),dev);},
    py::arg("n"),py::arg("dims"),py::arg("device")=0)

  .def_static("randn",[](const int n, const vector<int>& v, const int dev){
      return TensorPack<float>::gaussian(n,Gdims(v),dev);},
    py::arg("n"),py::arg("dims"),py::arg("device")=0)

//.def(pybind11::init<const at::Tensor&>())
//.def("torch",&Tensor<float>::torch)


// ---- I/O --------------------------------------------------------------------------------------------------


  .def("str",&TensorPack<float>::str,py::arg("indent")="")
  .def("__str__",&TensorPack<float>::str,py::arg("indent")="")
  .def("__repr__",&TensorPack<float>::str,py::arg("indent")="");

