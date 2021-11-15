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

  .def("str",&SO3vec::str,py::arg("indent")="")
  .def("__str__",&SO3vec::str,py::arg("indent")="")
  .def("__repr__",&SO3vec::repr,py::arg("indent")="");

