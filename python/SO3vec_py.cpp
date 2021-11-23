py::class_<SO3vec>(m,"SO3vec",
  "Class to store an SO(3)--covariant vector")

  .def_static("zero",static_cast<SO3vec (*)(const SO3type&)>(&SO3vec::zero))
  .def_static("ones",static_cast<SO3vec (*)(const SO3type&)>(&SO3vec::ones))
  .def_static("gaussian",static_cast<SO3vec (*)(const SO3type&)>(&SO3vec::gaussian))

  .def_static("zero",[](const vector<int>& v){
      return SO3vec::zero(SO3type(v));})
  .def_static("ones",[](const vector<int>& v){
      return SO3vec::ones(SO3type(v));})
  .def_static("gaussian",[](const vector<int>& v){
      return SO3vec::gaussian(SO3type(v));})

  .def("__len__",&SO3vec::size)
  .def("type",&SO3vec::type)

  .def("__getitem__",&SO3vec::get_part)
  .def("__setitem__",[](SO3vec& obj, const int l, const SO3part& x){
      obj.set_part(l,x);})

  .def("__add__",[](const SO3vec& x, const SO3vec& y){return x+y;})
  .def("__sub__",[](const SO3vec& x, const SO3vec& y){return x-y;})
  .def("__mul__",[](const SO3vec& x, const float c){return x*c;})
  .def("__rmul__",[](const SO3vec& x, const float c){return x*c;})
    
  .def("__iadd__",[](SO3vec& x, const SO3vec& y){x+=y; return x;})
  .def("__isub__",[](SO3vec& x, const SO3vec& y){x+=y; return x;})
  
  .def("to",&SO3vec::to_device)

  .def("str",&SO3vec::str,py::arg("indent")="")
  .def("__str__",&SO3vec::str,py::arg("indent")="")
  .def("__repr__",&SO3vec::repr,py::arg("indent")="");


m.def("inp",[](const SO3vec& x, const SO3vec& y){return inp(x,y).get_value();});
//m.def("odot",[](const CtensorObj& x, const CtensorObj& y){return x.odot(y);});
m.def("norm2",[](const SO3vec& x){return norm2(x).get_value();});

m.def("CGproduct",[](const SO3vec& x, const SO3vec& y){return CGproduct(x,y);});
