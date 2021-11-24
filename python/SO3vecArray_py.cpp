pybind11::class_<SO3vecArray>(m,"SO3vecArr",
  "Class to store an array of SO3vec objects.")

//.def_static("zero",static_cast<SO3vecArray (*)(const Gdims&, const int, const int)>(&SO3vecArray::zero))
  .def_static("zero",[](const Gdims& adims, const SO3type& tau, const int dev){
      return SO3vecArray::zero(adims,tau,-1,dev);}, 
    py::arg("adims"), py::arg("type"), py::arg("device")=0)
//.def_static("zero",[](const vector<int>& av, const int l, int n, const int dev){
//    return SO3vecArray::zero(Gdims(av),l,n,-1,dev);},
//  py::arg("adims"),py::arg("l"),py::arg("n"),py::arg("device")=0)

//.def_static("ones",static_cast<SO3vecArray (*)(const Gdims&, const Gdims&, const int, const int)>(&SO3vecArray::ones))
  .def_static("ones",[](const Gdims& adims, const SO3type& tau, const int dev){
      return SO3vecArray::ones(adims,tau,dev);}, 
    py::arg("adims"),py::arg("type"),py::arg("device")=0)
//  .def_static("ones",[](const vector<int>& av, const int l, const int n, const int dev){
//      return SO3vecArray::ones(Gdims(av),l,n,-1,dev);},
//    py::arg("adims"),py::arg("l"),py::arg("n"),py::arg("device")=0)

//.def_static("gaussian",static_cast<SO3vecArray (*)(const Gdims&, const Gdims&, const int, const int)>(&SO3vecArray::gaussian))
  .def_static("gaussian",[](const Gdims& adims, const SO3type& tau, const int dev){
      return SO3vecArray::gaussian(adims,tau,-1,dev);}, 
    py::arg("adims"),py::arg("type"),py::arg("device")=0)
//  .def_static("gaussian",[](const vector<int>& av, const int l, const int n, const int dev){
//      return SO3vecArray::gaussian(Gdims(av),l,n,-1,dev);},
//    py::arg("adims"),py::arg("l"),py::arg("n"),py::arg("device")=0)


  .def("__len__",&SO3vec::size)
  .def("type",&SO3vec::type)

  .def("get_adims",&CtensorArray::get_adims)
  .def("get_adim",&CtensorArray::get_adim)

  .def("get_cell",[](const SO3vecArray& obj, const Gindex& ix){
      return SO3vecArray(obj.get_cell(ix));})
  .def("get_cell",[](const SO3vecArray& obj, const vector<int> v){
      return SO3vecArray(obj.get_cell(Gindex(v)));})
  .def("__call__",[](const SO3vecArray& obj, const Gindex& ix){
      return SO3vecArray(obj.get_cell(ix));})
  .def("__call__",[](const SO3vecArray& obj, const vector<int> v){
      return SO3vecArray(obj.get_cell(Gindex(v)));})
  .def("__getitem__",[](const SO3vecArray& obj, const Gindex& ix){
      return SO3vecArray(obj.get_cell(ix));})
  .def("__getitem__",[](const SO3vecArray& obj, const vector<int> v){return obj.get_cell(Gindex(v));})

  .def("__setitem__",[](SO3vecArray& obj, const Gindex& ix, const SO3vec& x){
      obj.set_cell(ix,x);})
  .def("__setitem__",[](SO3vecArray& obj, const vector<int> v, const SO3vec& x){
      obj.set_cell(Gindex(v),x);})

  .def("__add__",[](const SO3vecArray& x, const SO3vecArray& y){
      return SO3vecArray(x.plus(y));})
  .def("__sub__",[](const SO3vecArray& x, const SO3vecArray& y){
      return SO3vecArray(x-y);})
  .def("__mul__",[](const SO3vecArray& x, const float c){
      SO3vecArray R(x.get_adims(),x.getl(),x.getn(),x.get_nbu(),fill::zero,x.get_dev());
      R.add(x,c);
      return R;})
  .def("__rmul__",[](const SO3vecArray& x, const float c){
      SO3vecArray R(x.get_dims(),x.getl(),x.getn(),x.get_nbu(),fill::zero,x.get_dev());
      R.add(x,c);
      return R;})
//.def("__mul__",[](const SO3vecArray& x, const SO3vecArray& y){
//    SO3vecArray R(x.get_dims().Mprod(y.get_dims()),x.get_nbu(),fill::zero,x.get_dev());
//    R.add_mprod(x,y);
//    return R;})

  .def("__iadd__",[](SO3vecArray& x, const SO3vecArray& y){x.add(y); return x;})
  .def("__isub__",[](SO3vecArray& x, const SO3vecArray& y){x.subtract(y); return x;})

//.def("__add__",[](const SO3vecArray& x, const SO3vec& y){return x.broadcast_plus(y);})

//.def("widen",&SO3vecArray::widen)
//.def("reduce",&SO3vecArray::reduce)

  .def("to",&SO3vecArray::to_device)

  .def("str",&SO3vecArray::str,py::arg("indent")="")
  .def("__str__",&SO3vecArray::str,py::arg("indent")="");


// ---- Stand-alone functions --------------------------------------------------------------------------------


//m.def("inp",[](const SO3vecArray& x, const SO3vecArray& y){return x.inp(y);});
//m.def("odot",[](const CtensorObj& x, const CtensorObj& y){return x.odot(y);});
//m.def("norm2",[](const SO3vecArray& x){return x.norm2();});

//m.def("inp",[](const SO3vecArray& x, const SO3vecArray& y){return x.inp(y);});

m.def("CGproduct",[](const SO3vecArray& x, const SO3vecArray& y){return CGproduct(x,y);});

