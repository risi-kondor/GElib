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


pybind11::class_<RtensorObj>(m,"rtensor")

  .def(pybind11::init<const Gdims&>())
  .def(pybind11::init<const Gdims&, const fill_raw&>())
  .def(pybind11::init<const Gdims&, const fill_zero&>())
  .def(pybind11::init<const Gdims&, const fill_ones&>())
  .def(pybind11::init<const Gdims&, const fill_identity&>())
  .def(pybind11::init<const Gdims&, const fill_gaussian&>())
  .def(pybind11::init<const Gdims&, const fill_sequential&>())

  .def(pybind11::init<const at::Tensor&>())

  .def_static("view",static_cast<RtensorObj(*)(at::Tensor&)>(&RtensorObj::view))
  .def_static("is_viewable",static_cast<bool(*)(const at::Tensor&)>(&RtensorObj::is_viewable))
  .def("torch",&RtensorObj::torch)

  .def_static("raw",static_cast<RtensorObj (*)(const Gdims&, const int, const int)>(&RtensorObj::raw))
  .def_static("raw",[](const Gdims& dims, const int dev){return RtensorObj::raw(dims,-1,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("raw",[](const vector<int>& v, const int dev){return RtensorObj::raw(Gdims(v),-1,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("raw",[](const int i0){return RtensorObj::raw(Gdims({i0}));})
  .def_static("raw",[](const int i0, const int i1){return RtensorObj::raw(Gdims({i0,i1}));})
  .def_static("raw",[](const int i0, const int i1, const int i2){return RtensorObj::raw(Gdims({i0,i1,i2}));})

  .def_static("zero",static_cast<RtensorObj (*)(const Gdims&, const int, const int)>(&RtensorObj::zero))
  .def_static("zero",[](const Gdims& dims, const int dev){return RtensorObj::zero(dims,-1,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("zero",[](const vector<int>& v, const int dev){return RtensorObj::zero(Gdims(v),-1,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("zero",[](const int i0){return RtensorObj::zero(Gdims({i0}));})
  .def_static("zero",[](const int i0, const int i1){return RtensorObj::zero(Gdims({i0,i1}));})
  .def_static("zero",[](const int i0, const int i1, const int i2){return RtensorObj::zero(Gdims({i0,i1,i2}));})

  .def_static("zeros",static_cast<RtensorObj (*)(const Gdims&, const int, const int)>(&RtensorObj::zero))
  .def_static("zeros",[](const Gdims& dims, const int dev){return RtensorObj::zero(dims,-1,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("zeros",[](const vector<int>& v, const int dev){return RtensorObj::zero(Gdims(v),-1,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("zeros",[](const int i0){return RtensorObj::zero(Gdims({i0}));})
  .def_static("zeros",[](const int i0, const int i1){return RtensorObj::zero(Gdims({i0,i1}));})
  .def_static("zeros",[](const int i0, const int i1, const int i2){return RtensorObj::zero(Gdims({i0,i1,i2}));})

  .def_static("ones",static_cast<RtensorObj (*)(const Gdims&,const int, const int)>(&RtensorObj::ones))
  .def_static("ones",[](const Gdims& dims, const int dev){return RtensorObj::ones(dims,-1,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("ones",[](const vector<int>& v, const int dev){return RtensorObj::ones(Gdims(v),-1,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("ones",[](const int i0){return RtensorObj::ones(Gdims({i0}));})
  .def_static("ones",[](const int i0, const int i1){return RtensorObj::ones(Gdims({i0,i1}));})
  .def_static("ones",[](const int i0, const int i1, const int i2){return RtensorObj::ones(Gdims({i0,i1,i2}));})

  .def_static("identity",static_cast<RtensorObj (*)(const Gdims&,const int, const int)>(&RtensorObj::identity))
  .def_static("identity",[](const Gdims& dims, const int dev){return RtensorObj::identity(dims,-1,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("identity",[](const vector<int>& v, const int dev){return RtensorObj::identity(Gdims(v),-1,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("identity",[](const int i0){return RtensorObj::identity(Gdims({i0}));})
  .def_static("identity",[](const int i0, const int i1){return RtensorObj::identity(Gdims({i0,i1}));})
  .def_static("identity",[](const int i0, const int i1, const int i2){return RtensorObj::identity(Gdims({i0,i1,i2}));})

  .def_static("gaussian",static_cast<RtensorObj (*)(const Gdims&,const int, const int)>(&RtensorObj::gaussian))
  .def_static("gaussian",[](const Gdims& dims, const int dev){return RtensorObj::gaussian(dims,-1,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("gaussian",[](const vector<int>& v, const int dev){return RtensorObj::gaussian(Gdims(v),-1,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("gaussian",[](const int i0){return RtensorObj::gaussian(Gdims({i0}));})
  .def_static("gaussian",[](const int i0, const int i1){return RtensorObj::gaussian(Gdims({i0,i1}));})
  .def_static("gaussian",[](const int i0, const int i1, const int i2){return RtensorObj::gaussian(Gdims({i0,i1,i2}));})

  .def_static("randn",static_cast<RtensorObj (*)(const Gdims&,const int, const int)>(&RtensorObj::gaussian))
  .def_static("randn",[](const Gdims& dims, const int dev){return RtensorObj::gaussian(dims,-1,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("randn",[](const vector<int>& v, const int dev){return RtensorObj::gaussian(Gdims(v),-1,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("randn",[](const int i0){return RtensorObj::gaussian(Gdims({i0}));})
  .def_static("randn",[](const int i0, const int i1){return RtensorObj::gaussian(Gdims({i0,i1}));})
  .def_static("randn",[](const int i0, const int i1, const int i2){return RtensorObj::gaussian(Gdims({i0,i1,i2}));})

  .def_static("sequential",static_cast<RtensorObj (*)(const Gdims&,const int, const int)>(&RtensorObj::sequential))
  .def_static("sequential",[](const Gdims& dims, const int dev){return RtensorObj::sequential(dims,-1,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("sequential",[](const vector<int>& v, const int dev){return RtensorObj::sequential(Gdims(v),-1,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("sequential",[](const int i0){return RtensorObj::sequential(Gdims({i0}));})
  .def_static("sequential",[](const int i0, const int i1){return RtensorObj::sequential(Gdims({i0,i1}));})
  .def_static("sequential",[](const int i0, const int i1, const int i2){return RtensorObj::sequential(Gdims({i0,i1,i2}));})

  .def("copy",[](const RtensorObj& obj){return obj;})


// ---- Conversions, transport, etc. ------------------------------------------------------------------------


  .def("add_to_grad",&RtensorObj::add_to_grad)
  .def("get_grad",&RtensorObj::get_grad)
  .def("view_of_grad",&RtensorObj::view_of_grad)

  .def("device",&RtensorObj::get_device)
  .def("to",&RtensorObj::to_device)
  .def("to_device",&RtensorObj::to_device)
  .def("move_to",[](RtensorObj& x, const int _dev){x.move_to_device(_dev);})


// ---- Access ---------------------------------------------------------------------------------------------- 


  .def("get_k",&RtensorObj::get_k)
  .def("getk",&RtensorObj::get_k)
  .def("get_ndims",&RtensorObj::get_k)
  .def("get_dims",&RtensorObj::get_dims)
  .def("get_dim",&RtensorObj::get_dim)
  .def("ndims",&RtensorObj::get_k)
  .def("dims",&RtensorObj::get_dims)
  .def("dim",&RtensorObj::get_dim)

  .def("get",static_cast<RscalarObj(RtensorObj::*)(const Gindex& )const>(&RtensorObj::get))
  .def("get",static_cast<RscalarObj(RtensorObj::*)(const int)const>(&RtensorObj::get))
  .def("get",static_cast<RscalarObj(RtensorObj::*)(const int, const int)const>(&RtensorObj::get))
  .def("get",static_cast<RscalarObj(RtensorObj::*)(const int, const int, const int)const>(&RtensorObj::get))

  .def("set",static_cast<RtensorObj&(RtensorObj::*)(const Gindex&, const RscalarObj&)>(&RtensorObj::set))
  .def("set",static_cast<RtensorObj&(RtensorObj::*)(const int, const RscalarObj&)>(&RtensorObj::set))
  .def("set",static_cast<RtensorObj&(RtensorObj::*)(const int, const int, const RscalarObj&)>(&RtensorObj::set))
  .def("set",static_cast<RtensorObj&(RtensorObj::*)(const int, const int, const int, const RscalarObj&)>(&RtensorObj::set))

  .def("get_value",static_cast<float(RtensorObj::*)(const Gindex& )const>(&RtensorObj::get_value))
  .def("get_value",static_cast<float(RtensorObj::*)(const int)const>(&RtensorObj::get_value))
  .def("get_value",static_cast<float(RtensorObj::*)(const int, const int)const>(&RtensorObj::get_value))
  .def("get_value",static_cast<float(RtensorObj::*)(const int, const int, const int)const>(&RtensorObj::get_value))

  .def("__call__",static_cast<float(RtensorObj::*)(const Gindex& )const>(&RtensorObj::get_value))
  .def("__call__",[](const RtensorObj& obj, const vector<int> v){return obj.get_value(Gindex(v));})
  .def("__getitem__",static_cast<float(RtensorObj::*)(const Gindex& )const>(&RtensorObj::get_value))
  .def("__getitem__",[](const RtensorObj& obj, const vector<int> v){return obj.get_value(Gindex(v));})

  .def("__call__",[](const RtensorObj& obj, const int i0){return obj.get_value(i0);})
  .def("__call__",[](const RtensorObj& obj, const int i0, const int i1){return obj.get_value(i0,i1);})
  .def("__call__",[](const RtensorObj& obj, const int i0, const int i1, const int i2){
      return obj.get_value(i0,i1,i2);})

  .def("set_value",static_cast<RtensorObj&(RtensorObj::*)(const Gindex&, const float)>(&RtensorObj::set_value))
  .def("set_value",static_cast<RtensorObj&(RtensorObj::*)(const int, const float)>(&RtensorObj::set_value))
  .def("set_value",static_cast<RtensorObj&(RtensorObj::*)(const int, const int, const float)>(&RtensorObj::set_value))
  .def("set_value",static_cast<RtensorObj&(RtensorObj::*)(const int, const int, const int, const float)>(&RtensorObj::set_value))

  .def("__setitem__",[](RtensorObj& obj, const Gindex& ix, const float x){obj.set_value(ix,x);})
  .def("__setitem__",[](RtensorObj& obj, const vector<int> v, const float x){obj.set_value(Gindex(v),x);})


// ---- Cumulative operations --------------------------------------------------------------------------------


  .def("add",[](RtensorObj& x, const RtensorObj& y, const float c){x.add(y,c);})


// ---- Operations -------------------------------------------------------------------------------------------


  .def("__iadd__",[](RtensorObj& x, const RtensorObj& y){x.add(y); return x;})
  .def("__isub__",[](RtensorObj& x, const RtensorObj& y){x.subtract(y); return x;})

  .def("__add__",[](const RtensorObj& x, const RtensorObj& y){return x.plus(y);})
  .def("__sub__",[](const RtensorObj& x, const RtensorObj& y){return x-y;})
  .def("__mul__",[](const RtensorObj& x, const float c){
      RtensorObj R(x.get_dims(),x.get_nbu(),fill::zero,x.get_dev());
      R.add(x,c);
      return R;})
  .def("__rmul__",[](const RtensorObj& x, const float c){
      RtensorObj R(x.get_dims(),x.get_nbu(),fill::zero,x.get_dev());
      R.add(x,c);
      return R;})
  .def("__mul__",[](const RtensorObj& x, const RtensorObj& y){
      RtensorObj R(x.get_dims().Mprod(y.get_dims()),x.get_nbu(),fill::zero,x.get_dev());
      R.add_mprod(x,y);
      return R;})

  .def("slice",&RtensorObj::slice)
  .def("chunk",&RtensorObj::chunk)

  .def("reshape",&RtensorObj::reshape)
  .def("reshape",[](RtensorObj& obj, const vector<int>& v){obj.reshape(Gindex(v));})

  .def("transp",&RtensorObj::transp)

//.def("norm2",static_cast<float(RtensorObj::*)() const>(&RtensorObj::norm2))
//.def("inp",static_cast<float(RtensorObj::*)(const RtensorObj&) const>(&RtensorObj::inp))
//.def("diff2",static_cast<float(RtensorObj::*)(const RtensorObj&) const>(&RtensorObj::diff2))

  .def("norm2",&RtensorObj::norm2)
  .def("inp",[](const RtensorObj& x, const RtensorObj& y){return x.inp(y);})
  .def("diff2",[](const RtensorObj& x, const RtensorObj& y){return x.diff2(y);})

//.def("__getitem__",static_cast<CscalarObj(RtensorObj::*)(const int, const int)const>(&RtensorObj::get))


// ---- I/O --------------------------------------------------------------------------------------------------


  .def("str",[](const RtensorObj& x){return x.str();})
  .def("__str__",[](const RtensorObj& x){return x.str();})
  .def("__repr__",[](const RtensorObj& x){return x.str();})
;



m.def("inp",[](const RtensorObj& x, const RtensorObj& y){return x.inp(y);});
m.def("odot",[](const RtensorObj& x, const RtensorObj& y){return x.inp(y);});
m.def("norm2",[](const RtensorObj& x){return x.norm2();});

m.def("ReLU",[](const RtensorObj& x){
    RtensorObj R(x.get_dims(),x.get_nbu(),fill::zero);
    R.add_ReLU(x);
    return R;});
m.def("ReLU",[](const RtensorObj& x, const float c){
    RtensorObj R(x.get_dims(),x.get_nbu(),fill::zero);
    R.add_ReLU(x,c);
    return R;});

