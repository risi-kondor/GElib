

  py::class_<SO3type>(m,"SO3type","Class to store the type of an SO3-vector")

    .def(pybind11::init<>(),"")
    .def(pybind11::init<vector<int> >(),"")
    
    .def("__len__",&SO3type::size)
    .def("maxl",&SO3type::maxl)
    .def("__getitem__",&SO3type::operator())
    .def("__setitem__",&SO3type::set)
    

    .def("str",&SO3type::str,py::arg("indent")="","Print the SO3type to string.")
    .def("__str__",&SO3type::str,py::arg("indent")="","Print the SO3type to string.")
    .def("__repr__",&SO3type::repr,py::arg("indent")="","Print the SO3type to string.");


  m.def("CGproduct",static_cast<SO3type (*)(const SO3type&, const SO3type&, const int)>(&CGproduct),
    py::arg("x"),py::arg("y"),py::arg("maxl")=-1);


/*
  py::class_<SO3bitype>(m,"SO3bitype","Class to store the type of an SO3-bivector")

    .def(pybind11::init<>(),"")
    .def(pybind11::init<vector<vector<int> > >(),"")
    
    .def("__len__",&SO3type::size)
    .def("maxl",&SO3type::maxl)
    .def("__getitem__",&SO3type::operator())
    .def("__setitem__",&SO3type::set)
    

    .def("str",&SO3bitype::str,py::arg("indent")="","Print the SO3bitype to string.")
    .def("__str__",&SO3bitype::str,py::arg("indent")="","Print the SO3bitype to string.")
    .def("__repr__",&SO3bitype::repr,py::arg("indent")="","Print the SO3bitype to string.");

*/
