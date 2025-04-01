

py::class_<SO3type>(m,"SO3type","Class to store the type of an SO3-vector")

  .def(pybind11::init<>(),"")
  .def(pybind11::init<vector<int> >(),"")
  .def(pybind11::init<map<int,int> >(),"")
    
  .def("__len__",&SO3type::size)
  .def("maxl",&SO3type::highest)
  .def("__getitem__",&SO3type::operator())
  .def("__setitem__",&SO3type::set)
  .def("get_parts",[](const SO3type& x){return x.parts;})
    
  .def("CGproduct",[](const SO3type& x, const SO3type& y) {return x.CGproduct(y,-1);})
  .def("CGproduct",[](const SO3type& x, const SO3type& y, int maxl) {return x.CGproduct(y,maxl);})

  .def("DiagCGproduct",[](const SO3type& x, const SO3type& y) {return x.DiagCGproduct(y,-1);})
  .def("DiagCGproduct",[](const SO3type& x, const SO3type& y, int maxl) {return x.DiagCGproduct(y,maxl);})

  .def("str",&SO3type::str,py::arg("indent")="","Print the SO3type to string.")
  .def("__str__",&SO3type::str,py::arg("indent")="","Print the SO3type to string.")
  .def("__repr__",&SO3type::repr,"Print the SO3type to string.");


m.def("CGproduct",[](const SO3type& x, const SO3type& y){
    return CGproduct(x,y,-1);});

m.def("CGproduct",[](const SO3type& x, const SO3type& y, const int maxl){
    return CGproduct(x,y,maxl);});


