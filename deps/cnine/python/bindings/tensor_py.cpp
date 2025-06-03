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


pybind11::class_<Tensor<float> >(m,"tensor")

  .def_static("zero",[](const vector<int>& v, const int dev){
      return Tensor<float>::zero(Gdims(v),dev);},
    py::arg("dims"),py::arg("device")=0)

  .def_static("zeros",[](const vector<int>& v, const int dev){
      return Tensor<float>::zero(Gdims(v),dev);},
    py::arg("dims"),py::arg("device")=0)

  .def_static("sequential",[](const vector<int>& v, const int dev){
      return Tensor<float>::sequential(Gdims(v),dev);},
    py::arg("dims"),py::arg("device")=0)

  .def_static("randn",[](const vector<int>& v, const int dev){
      return Tensor<float>::gaussian(Gdims(v),dev);},
    py::arg("dims"),py::arg("device")=0)

  .def(pybind11::init<const at::Tensor&>())

  .def("torch",&Tensor<float>::torch)


// ---- I/O --------------------------------------------------------------------------------------------------


  .def("str",&Tensor<float>::str,py::arg("indent")="")
  .def("__str__",&Tensor<float>::str,py::arg("indent")="")
  .def("__repr__",&Tensor<float>::str,py::arg("indent")="");

