
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


pybind11::class_<CtensorB>(m,"ctensorb")

  .def(pybind11::init<const Gdims&>())
  .def(pybind11::init<const Gdims&, const fill_raw&>())
  .def(pybind11::init<const Gdims&, const fill_zero&>())
  .def(pybind11::init<const Gdims&, const fill_ones&>())
//.def(pybind11::init<const Gdims&, const fill_identity&>())
  .def(pybind11::init<const Gdims&, const fill_gaussian&>())
  .def(pybind11::init<const Gdims&, const fill_sequential&>())

  .def(pybind11::init<const at::Tensor&>())
  .def_static("view",static_cast<CtensorB(*)(at::Tensor&)>(&CtensorB::view))
//  .def_static("is_viewable",static_cast<bool(*)(const at::Tensor&)>(&CtensorB::is_viewable))
//.def_static("view",static_cast<CtensorB>(*)(const at::Tensor&)>(&CtensorB::view))
//.def_static("const_view",static_cast<CtensorB>(*)(at::Tensor&)>(&CtensorB::const_view))
  .def("torch",&CtensorB::torch)

  .def_static("raw",static_cast<CtensorB(*)(const Gdims&, const int)>(&CtensorB::raw))
  .def_static("raw",[](const Gdims& dims, const int dev){return CtensorB::raw(dims,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("raw",[](const vector<int>& v, const int dev){return CtensorB::raw(Gdims(v),dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("raw",[](const int i0){return CtensorB::raw(Gdims({i0}));})
  .def_static("raw",[](const int i0, const int i1){return CtensorB::raw(Gdims({i0,i1}));})
  .def_static("raw",[](const int i0, const int i1, const int i2){return CtensorB::raw(Gdims({i0,i1,i2}));})

  .def_static("zero",static_cast<CtensorB (*)(const Gdims&, const int)>(&CtensorB::zero))
  .def_static("zero",[](const Gdims& dims, const int dev){return CtensorB::zero(dims,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("zero",[](const vector<int>& v, const int dev){return CtensorB::zero(Gdims(v),dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("zero",[](const int i0){return CtensorB::zero(Gdims({i0}));})
  .def_static("zero",[](const int i0, const int i1){return CtensorB::zero(Gdims({i0,i1}));})
  .def_static("zero",[](const int i0, const int i1, const int i2){return CtensorB::zero(Gdims({i0,i1,i2}));})

  .def_static("ones",static_cast<CtensorB (*)(const Gdims&,const int)>(&CtensorB::ones))
  .def_static("ones",[](const Gdims& dims, const int dev){return CtensorB::ones(dims,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("ones",[](const vector<int>& v, const int dev){return CtensorB::ones(Gdims(v),dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("ones",[](const int i0){return CtensorB::ones(Gdims({i0}));})
  .def_static("ones",[](const int i0, const int i1){return CtensorB::ones(Gdims({i0,i1}));})
  .def_static("ones",[](const int i0, const int i1, const int i2){return CtensorB::ones(Gdims({i0,i1,i2}));})

/*
  .def_static("identity",static_cast<CtensorB (*)(const Gdims&,const int)>(&CtensorB::identity))
  .def_static("identity",[](const Gdims& dims, const int dev){return CtensorB::identity(dims,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("identity",[](const vector<int>& v, const int dev){return CtensorB::identity(Gdims(v),dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("identity",[](const int i0){return CtensorB::identity(Gdims({i0}));})
  .def_static("identity",[](const int i0, const int i1){return CtensorB::identity(Gdims({i0,i1}));})
  .def_static("identity",[](const int i0, const int i1, const int i2){return CtensorB::identity(Gdims({i0,i1,i2}));})
*/

  .def_static("gaussian",static_cast<CtensorB (*)(const Gdims&,const int)>(&CtensorB::gaussian))
  .def_static("gaussian",[](const Gdims& dims, const int dev){return CtensorB::gaussian(dims,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("gaussian",[](const vector<int>& v, const int dev){return CtensorB::gaussian(Gdims(v),dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("gaussian",[](const int i0){return CtensorB::gaussian(Gdims({i0}));})
  .def_static("gaussian",[](const int i0, const int i1){return CtensorB::gaussian(Gdims({i0,i1}));})
  .def_static("gaussian",[](const int i0, const int i1, const int i2){return CtensorB::gaussian(Gdims({i0,i1,i2}));})

  .def_static("randn",static_cast<CtensorB (*)(const Gdims&,const int)>(&CtensorB::gaussian))
  .def_static("randn",[](const Gdims& dims, const int dev){return CtensorB::gaussian(dims,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("randn",[](const vector<int>& v, const int dev){return CtensorB::gaussian(Gdims(v),dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("randn",[](const int i0){return CtensorB::gaussian(Gdims({i0}));})
  .def_static("randn",[](const int i0, const int i1){return CtensorB::gaussian(Gdims({i0,i1}));})
  .def_static("randn",[](const int i0, const int i1, const int i2){return CtensorB::gaussian(Gdims({i0,i1,i2}));})

  .def_static("sequential",static_cast<CtensorB (*)(const Gdims&,const int)>(&CtensorB::sequential))
  .def_static("sequential",[](const Gdims& dims, const int dev){return CtensorB::sequential(dims,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("sequential",[](const vector<int>& v, const int dev){return CtensorB::sequential(Gdims(v),dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("sequential",[](const int i0){return CtensorB::sequential(Gdims({i0}));})
  .def_static("sequential",[](const int i0, const int i1){return CtensorB::sequential(Gdims({i0,i1}));})
  .def_static("sequential",[](const int i0, const int i1, const int i2){return CtensorB::sequential(Gdims({i0,i1,i2}));})

//.def("copy",[](const RtensorObj& obj){return obj;})

  .def("get_k",&CtensorB::getk)
  .def("getk",&CtensorB::getk)
  .def("get_dims",&CtensorB::get_dims)
  .def("get_dim",&CtensorB::get_dim)
  .def("dims",&CtensorB::get_dims)
  .def("dim",&CtensorB::get_dim)

/*
  .def("get",static_cast<CscalarObj(CtensorB::*)(const Gindex& )const>(&CtensorB::get))
  .def("get",static_cast<CscalarObj(CtensorB::*)(const int)const>(&CtensorB::get))
  .def("get",static_cast<CscalarObj(CtensorB::*)(const int, const int)const>(&CtensorB::get))
  .def("get",static_cast<CscalarObj(CtensorB::*)(const int, const int, const int)const>(&CtensorB::get))

  .def("set",static_cast<CtensorB&(CtensorB::*)(const Gindex&, const CscalarObj&)>(&CtensorB::set))
  .def("set",static_cast<CtensorB&(CtensorB::*)(const int, const CscalarObj&)>(&CtensorB::set))
  .def("set",static_cast<CtensorB&(CtensorB::*)(const int, const int, const CscalarObj&)>(&CtensorB::set))
  .def("set",static_cast<CtensorB&(CtensorB::*)(const int, const int, const int, const CscalarObj&)>(&CtensorB::set))

  .def("get_value",static_cast<complex<float>(CtensorB::*)(const Gindex& )const>(&CtensorB::get_value))
  .def("get_value",static_cast<complex<float>(CtensorB::*)(const int)const>(&CtensorB::get_value))
  .def("get_value",static_cast<complex<float>(CtensorB::*)(const int, const int)const>(&CtensorB::get_value))
  .def("get_value",static_cast<complex<float>(CtensorB::*)(const int, const int, const int)const>(&CtensorB::get_value))

  .def("__call__",static_cast<complex<float>(CtensorB::*)(const Gindex& )const>(&CtensorB::get_value))
  .def("__call__",[](const CtensorB& obj, const vector<int> v){return obj.get_value(Gindex(v));})

  .def("__call__",[](const CtensorB& obj, const int i0){return obj.get_value(i0);})
  .def("__call__",[](const CtensorB& obj, const int i0, const int i1){return obj.get_value(i0,i1);})
  .def("__call__",[](const CtensorB& obj, const int i0, const int i1, const int i2){
      return obj.get_value(i0,i1,i2);})

  .def("__getitem__",static_cast<complex<float>(CtensorB::*)(const Gindex& )const>(&CtensorB::get_value))
  .def("__getitem__",[](const CtensorB& obj, const vector<int> v){return obj.get_value(Gindex(v));})

  .def("set_value",static_cast<CtensorB&(CtensorB::*)(const Gindex&, const complex<float>)>(&CtensorB::set_value))
  .def("set_value",static_cast<CtensorB&(CtensorB::*)(const int, const complex<float>)>(&CtensorB::set_value))
  .def("set_value",static_cast<CtensorB&(CtensorB::*)(const int, const int, const complex<float>)>(&CtensorB::set_value))
  .def("set_value",static_cast<CtensorB&(CtensorB::*)(const int, const int, const int, const complex<float>)>(&CtensorB::set_value))

  .def("__setitem__",[](CtensorB& obj, const Gindex& ix, const complex<float> x){obj.set_value(ix,x);})
  .def("__setitem__",[](CtensorB& obj, const vector<int> v, const complex<float> x){obj.set_value(Gindex(v),x);})
*/


//.def("__getitem__",static_cast<CscalarObj(CtensorB::*)(const int, const int)const>(&CtensorB::get))

  .def("get_dev",&CtensorB::get_dev)
  .def("device",&CtensorB::get_device)
  .def("to",&CtensorB::to_device)
  .def("to_device",&CtensorB::to_device)
  .def("move_to",[](CtensorB& x, const int _dev){x.move_to_device(_dev);})

  .def("str",&CtensorB::str,py::arg("indent")="")
  .def("__str__",&CtensorB::str,py::arg("indent")="")
  .def("__repr__",&CtensorB::str,py::arg("indent")="");

