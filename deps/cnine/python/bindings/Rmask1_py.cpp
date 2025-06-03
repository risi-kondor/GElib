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


pybind11::class_<Rmask1>(m,"Rmask1")

  .def(pybind11::init([](at::Tensor& M){
      return Rmask1::matrix(RtensorA::view(M).view2());
      }))

  .def("inv",&Rmask1::inv)

  .def("__str__",&Rmask1::str,py::arg("indent")="")
  
  ;
