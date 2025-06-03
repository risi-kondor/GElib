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


pybind11::class_<CSRmatrix<float> >(m,"CSRmatrix")

  .def(pybind11::init<const int, const int>())

  .def(pybind11::init<const at::Tensor&>())
  .def("torch",&CSRmatrix<float>::torch)


// ---- Conversions, transport, etc. ------------------------------------------------------------------------

//.def("add_to_grad",&RtensorObj::add_to_grad)
//.def("get_grad",&RtensorObj::get_grad)
//.def("view_of_grad",&RtensorObj::view_of_grad)

  .def("device",&CSRmatrix<float>::get_device)
  .def("to_device",&CSRmatrix<float>::to_device)

// ---- I/O --------------------------------------------------------------------------------------------------


  .def("str",[](const CSRmatrix<float>& x){return x.str();})
  .def("__str__",[](const CSRmatrix<float>& x){return x.str();})
  .def("__repr__",[](const CSRmatrix<float>& x){return x.str();})
;
