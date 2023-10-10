
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#include <torch/torch.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>


// ---- cnine includes

#include "Cnine_base.cpp"
#include "/usr/local/include/Eigen/Dense"

// ---- Snob2 includes

#include "CombinatorialBank.hpp"
#include "SnBank.hpp"
namespace Snob2{
  CombinatorialBank* _combibank=new CombinatorialBank();
  SnBank* _snbank=new SnBank();
}


// ---- GElib includes

#include "GElib_base.cpp"
#include "GElibSession.hpp"
#include "SO3.hpp"

GElib::GElibSession session;

namespace GElib{
  SO3CouplingMatrices SO3::coupling_matrices;
  CGprodBasisBank<SO3> SO3::product_space_bank;
}


#include "CGprodBasis.hpp"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  using namespace GElib;

#include "CGprodBasis_py.cpp"


}
