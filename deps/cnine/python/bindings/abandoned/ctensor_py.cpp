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


pybind11::class_<CtensorObj>(m,"ctensor")

  .def(pybind11::init<const Gdims&>())
  .def(pybind11::init<const Gdims&, const fill_raw&>())
  .def(pybind11::init<const Gdims&, const fill_zero&>())
  .def(pybind11::init<const Gdims&, const fill_ones&>())
  .def(pybind11::init<const Gdims&, const fill_identity&>())
  .def(pybind11::init<const Gdims&, const fill_gaussian&>())
  .def(pybind11::init<const Gdims&, const fill_sequential&>())

  .def(pybind11::init<const at::Tensor&>())

//.def(pybind11::init<const int, const at::Tensor&>())

  .def_static("view",static_cast<CtensorObj(*)(at::Tensor&)>(&CtensorObj::view))
  .def("torch",&CtensorObj::torch)

  .def_static("raw",static_cast<CtensorObj (*)(const Gdims&, const int)>(&CtensorObj::raw))
  .def_static("raw",[](const Gdims& dims, const int dev){return CtensorObj::raw(dims,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("raw",[](const vector<int>& v, const int dev){return CtensorObj::raw(Gdims(v),dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("raw",[](const int i0){return CtensorObj::raw(Gdims({i0}));})
  .def_static("raw",[](const int i0, const int i1){return CtensorObj::raw(Gdims({i0,i1}));})
  .def_static("raw",[](const int i0, const int i1, const int i2){return CtensorObj::raw(Gdims({i0,i1,i2}));})

  .def_static("zero",static_cast<CtensorObj (*)(const Gdims&, const int)>(&CtensorObj::zero))
  .def_static("zero",[](const Gdims& dims, const int dev){return CtensorObj::zero(dims,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("zero",[](const vector<int>& v, const int dev){return CtensorObj::zero(Gdims(v),dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("zero",[](const int i0){return CtensorObj::zero(Gdims({i0}));})
  .def_static("zero",[](const int i0, const int i1){return CtensorObj::zero(Gdims({i0,i1}));})
  .def_static("zero",[](const int i0, const int i1, const int i2){return CtensorObj::zero(Gdims({i0,i1,i2}));})

  .def_static("zeros",static_cast<CtensorObj (*)(const Gdims&, const int)>(&CtensorObj::zero))
  .def_static("zeros",[](const Gdims& dims, const int dev){return CtensorObj::zero(dims,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("zeros",[](const vector<int>& v, const int dev){return CtensorObj::zero(Gdims(v),dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("zeros",[](const int i0){return CtensorObj::zero(Gdims({i0}));})
  .def_static("zeros",[](const int i0, const int i1){return CtensorObj::zero(Gdims({i0,i1}));})
  .def_static("zeros",[](const int i0, const int i1, const int i2){return CtensorObj::zero(Gdims({i0,i1,i2}));})

  .def_static("ones",static_cast<CtensorObj (*)(const Gdims&, const int)>(&CtensorObj::ones))
  .def_static("ones",[](const Gdims& dims, const int dev){return CtensorObj::ones(dims,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("ones",[](const vector<int>& v, const int dev){return CtensorObj::ones(Gdims(v),dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("ones",[](const int i0){return CtensorObj::ones(Gdims({i0}));})
  .def_static("ones",[](const int i0, const int i1){return CtensorObj::ones(Gdims({i0,i1}));})
  .def_static("ones",[](const int i0, const int i1, const int i2){return CtensorObj::ones(Gdims({i0,i1,i2}));})

  .def_static("identity",static_cast<CtensorObj (*)(const Gdims&, const int)>(&CtensorObj::identity))
  .def_static("identity",[](const Gdims& dims, const int dev){return CtensorObj::identity(dims,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("identity",[](const vector<int>& v, const int dev){return CtensorObj::identity(Gdims(v),dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("identity",[](const int i0){return CtensorObj::identity(Gdims({i0}));})
  .def_static("identity",[](const int i0, const int i1){return CtensorObj::identity(Gdims({i0,i1}));})
  .def_static("identity",[](const int i0, const int i1, const int i2){return CtensorObj::identity(Gdims({i0,i1,i2}));})

  .def_static("gaussian",static_cast<CtensorObj (*)(const Gdims&, const int)>(&CtensorObj::gaussian))
  .def_static("gaussian",[](const Gdims& dims, const int dev){return CtensorObj::gaussian(dims,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("gaussian",[](const vector<int>& v, const int dev){return CtensorObj::gaussian(Gdims(v),dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("gaussian",[](const int i0){return CtensorObj::gaussian(Gdims({i0}));})
  .def_static("gaussian",[](const int i0, const int i1){return CtensorObj::gaussian(Gdims({i0,i1}));})
  .def_static("gaussian",[](const int i0, const int i1, const int i2){return CtensorObj::gaussian(Gdims({i0,i1,i2}));})

  .def_static("randn",static_cast<CtensorObj (*)(const Gdims&, const int)>(&CtensorObj::gaussian))
  .def_static("randn",[](const Gdims& dims, const int dev){return CtensorObj::gaussian(dims,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("randn",[](const vector<int>& v, const int dev){return CtensorObj::gaussian(Gdims(v),dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("randn",[](const int i0){return CtensorObj::gaussian(Gdims({i0}));})
  .def_static("randn",[](const int i0, const int i1){return CtensorObj::gaussian(Gdims({i0,i1}));})
  .def_static("randn",[](const int i0, const int i1, const int i2){return CtensorObj::gaussian(Gdims({i0,i1,i2}));})

  .def_static("sequential",static_cast<CtensorObj (*)(const Gdims&, const int)>(&CtensorObj::sequential))
  .def_static("sequential",[](const Gdims& dims, const int dev){return CtensorObj::sequential(dims,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("sequential",[](const vector<int>& v, const int dev){return CtensorObj::sequential(Gdims(v),dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("sequential",[](const int i0){return CtensorObj::sequential(Gdims({i0}));})
  .def_static("sequential",[](const int i0, const int i1){return CtensorObj::sequential(Gdims({i0,i1}));})
  .def_static("sequential",[](const int i0, const int i1, const int i2){return CtensorObj::sequential(Gdims({i0,i1,i2}));})

  .def("copy",[](const RtensorObj& obj){return obj;})


// ---- Conversions, transport, etc. ------------------------------------------------------------------------


  .def("add_to_grad",&CtensorObj::add_to_grad)
  .def("get_grad",&CtensorObj::get_grad)
  .def("view_of_grad",&CtensorObj::view_of_grad)

  .def("device",&CtensorObj::get_device)
  .def("to",&CtensorObj::to_device)
  .def("to_device",&CtensorObj::to_device)
  .def("move_to",[](CtensorObj& x, const int _dev){x.move_to_device(_dev);})


// ---- Access ---------------------------------------------------------------------------------------------- 


  .def("get_k",&CtensorObj::get_k)
  .def("getk",&CtensorObj::get_k)
  .def("get_ndims",&CtensorObj::get_dims)
  .def("get_dims",&CtensorObj::get_dims)
  .def("get_dim",&CtensorObj::get_dim)
  .def("ndims",&CtensorObj::get_dims)
  .def("dims",&CtensorObj::get_dims)
  .def("dim",&CtensorObj::get_dim)

  .def("get",static_cast<CscalarObj(CtensorObj::*)(const Gindex& )const>(&CtensorObj::get))
  .def("get",static_cast<CscalarObj(CtensorObj::*)(const int)const>(&CtensorObj::get))
  .def("get",static_cast<CscalarObj(CtensorObj::*)(const int, const int)const>(&CtensorObj::get))
  .def("get",static_cast<CscalarObj(CtensorObj::*)(const int, const int, const int)const>(&CtensorObj::get))

  .def("set",static_cast<CtensorObj&(CtensorObj::*)(const Gindex&, const CscalarObj&)>(&CtensorObj::set))
  .def("set",static_cast<CtensorObj&(CtensorObj::*)(const int, const CscalarObj&)>(&CtensorObj::set))
  .def("set",static_cast<CtensorObj&(CtensorObj::*)(const int, const int, const CscalarObj&)>(&CtensorObj::set))
  .def("set",static_cast<CtensorObj&(CtensorObj::*)(const int, const int, const int, const CscalarObj&)>(&CtensorObj::set))

  .def("get_value",static_cast<complex<float>(CtensorObj::*)(const Gindex& )const>(&CtensorObj::get_value))
  .def("get_value",static_cast<complex<float>(CtensorObj::*)(const int)const>(&CtensorObj::get_value))
  .def("get_value",static_cast<complex<float>(CtensorObj::*)(const int, const int)const>(&CtensorObj::get_value))
  .def("get_value",static_cast<complex<float>(CtensorObj::*)(const int, const int, const int)const>(&CtensorObj::get_value))

  .def("__call__",static_cast<complex<float>(CtensorObj::*)(const Gindex& )const>(&CtensorObj::get_value))
  .def("__call__",[](const CtensorObj& obj, const vector<int> v){return obj.get_value(Gindex(v));})

  .def("__call__",[](const CtensorObj& obj, const int i0){return obj.get_value(i0);})
  .def("__call__",[](const CtensorObj& obj, const int i0, const int i1){return obj.get_value(i0,i1);})
  .def("__call__",[](const CtensorObj& obj, const int i0, const int i1, const int i2){
      return obj.get_value(i0,i1,i2);})

  .def("__getitem__",static_cast<complex<float>(CtensorObj::*)(const Gindex& )const>(&CtensorObj::get_value))
  .def("__getitem__",[](const CtensorObj& obj, const vector<int> v){return obj.get_value(Gindex(v));})

  .def("set_value",static_cast<CtensorObj&(CtensorObj::*)(const Gindex&, const complex<float>)>(&CtensorObj::set_value))
  .def("set_value",static_cast<CtensorObj&(CtensorObj::*)(const int, const complex<float>)>(&CtensorObj::set_value))
  .def("set_value",static_cast<CtensorObj&(CtensorObj::*)(const int, const int, const complex<float>)>(&CtensorObj::set_value))
  .def("set_value",static_cast<CtensorObj&(CtensorObj::*)(const int, const int, const int, const complex<float>)>(&CtensorObj::set_value))

  .def("__setitem__",[](CtensorObj& obj, const Gindex& ix, const complex<float> x){obj.set_value(ix,x);})
  .def("__setitem__",[](CtensorObj& obj, const vector<int> v, const complex<float> x){obj.set_value(Gindex(v),x);})

// ---- Cumulative operations --------------------------------------------------------------------------------


  .def("add",[](CtensorObj& x, const CtensorObj& y, const float c){x.add(y,c);})
  .def("add_conj",[](CtensorObj& x, const CtensorObj& y, const float c){x.add(y,c);})


// ---- Operations -------------------------------------------------------------------------------------------

  .def("__add__",[](const CtensorObj& x, const CtensorObj& y){return x.plus(y);})
  .def("__subtract__",[](const CtensorObj& x, const CtensorObj& y){return x-y;})
  .def("__mul__",[](const CtensorObj& x, const complex<float> c){
      CtensorObj R(x.get_dims(),x.get_nbu(),fill::zero,x.get_dev());
      R.add(x,c);
      return R;})
  .def("__mul__",[](const complex<float> c, const CtensorObj& x){
      CtensorObj R(x.get_dims(),x.get_nbu(),fill::zero,x.get_dev());
      R.add(x,c);
      return R;})
  .def("__mul__",[](const CtensorObj& x, const CtensorObj& y){
      CtensorObj R(x.get_dims().Mprod(y.get_dims()),x.get_nbu(),fill::zero,x.get_dev());
      R.add_mprod(x,y);
      return R;})

  .def("slice",&CtensorObj::slice)
  .def("chunk",&CtensorObj::chunk)

//.def("reshape",&CtensorObj::reshape)
//.def("reshape",[](CtensorObj& obj, const vector<int>& v){obj.reshape(Gindex(v));})

  .def("transp",&CtensorObj::transp)
  .def("conj",&CtensorObj::conj)
  .def("herm",&CtensorObj::herm)

//.def("__getitem__",static_cast<CscalarObj(CtensorObj::*)(const int, const int)const>(&CtensorObj::get))

  .def("norm2",&CtensorObj::norm2)
  .def("inp",[](const CtensorObj& x, const CtensorObj& y){return x.inp(y);})
  .def("diff2",[](const CtensorObj& x, const CtensorObj& y){return x.diff2(y);})


// ---- I/O --------------------------------------------------------------------------------------------------


  .def("str",&CtensorObj::str,py::arg("indent")="")
  .def("__str__",&CtensorObj::str,py::arg("indent")="")
  .def("__repr__",&CtensorObj::str,py::arg("indent")="");



m.def("inp",[](const CtensorObj& x, const CtensorObj& y){return x.inp(y);});
//m.def("odot",[](const CtensorObj& x, const CtensorObj& y){return x.odot(y);});
m.def("norm2",[](const CtensorObj& x){return x.norm2();});


m.def("ReLU",[](const CtensorObj& x){
    CtensorObj R(x.get_dims(),x.get_nbu(),fill::zero);
    R.add_ReLU(x);
    return R;});
m.def("ReLU",[](const CtensorObj& x, const float c){
    CtensorObj R(x.get_dims(),x.get_nbu(),fill::zero);
    R.add_ReLU(x,c);
    return R;});

