/*
 * This file is part of GElib, a C++/CUDA library for group equivariant 
 * tensor operations. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with GElib in the file NONCOMMERICAL.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in orginal
 * or modified form) must retain this copyright notice and must be 
 * accompanied by a verbatim copy of the license. 
 *
 */


#include <torch/torch.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include "GElib_base.cpp"
#include "GElibSession.hpp"

#include "GatherMapB.hpp" // temporary 

#include "Gpart.hpp"
#include "Gvec.hpp"
#include "CGproduct.hpp"
#include "DiagCGproduct.hpp"

#include "SO3element.hpp"
#include "SO3irrep.hpp"
#include "SO3type.hpp"
#include "SO3part.hpp"
#include "SO3vec.hpp"

#include "O3element.hpp"
#include "O3irrep.hpp"
#include "O3type.hpp"
#include "O3part.hpp"
#include "O3vec.hpp"


GElib::GElibSession session;


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {


  using namespace cnine;
  using namespace GElib;
  namespace py=pybind11;

  py::options options;

  m.def("version",[](){cout<<_GELIB_VERSION<<endl;});

  typedef cnine::TensorView<float> tensorf;
  typedef cnine::TensorView<complex<float> > tensorc;

  #include "gather_map_py.cpp"

  #include "SO3element_py.cpp"
  #include "SO3irrep_py.cpp"
  #include "SO3type_py.cpp"
  #include "SO3part_py.cpp"
  #include "SO3vec_py.cpp"

  #include "O3element_py.cpp"
  #include "O3irrep_py.cpp"
  #include "O3type_py.cpp"
  #include "O3part_py.cpp"
  #include "O3vec_py.cpp"

}

