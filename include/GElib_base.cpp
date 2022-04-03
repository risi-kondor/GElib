
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#include "Cnine_base.cpp"
#include "GElib_base.hpp"

#ifdef _WITH_CENGINE
#include "Cengine_base.cpp"
#endif 

#include "SO3_CGbank.hpp"
#include "SO3_SPHgen.hpp"
#include "Factorial.hpp"
//#include "SO3CGprogramBank.hpp"


/*
#include "SO3part_ops.hpp"
#include "SO3vec_ops.hpp"
#include "SO3vec_add_ops.hpp"
#include "SO3vec_add_inp_ops.hpp"
#include "SO3vec_CGproduct_ops.hpp"
*/

vector<int> GElib::factorial::fact;

GElib::SO3_CGbank SO3_cgbank;
GElib::SO3_SPHgen SO3_sphGen;

//GElib::SO3CGprogramBank SO3_CGprogram_bank;

namespace GElib{

  class CombinatorialBank;
  CombinatorialBank* _combibank=nullptr;

  class SnBank;
  SnBank* _snbank=nullptr;

}

//GElib::SnObjects* _Snobjects;

/*
GElib::Dprogram GElib_global_program;

#include "GElibSession.hpp"

int GElib::SO3part_add_CGproduct_op::_batcher_id=0; 
int GElib::SO3part_add_CGproduct_back0_op::_batcher_id=0; 
int GElib::SO3part_add_CGproduct_back1_op::_batcher_id=0; 

int GElib::SO3part_add_CGproduct_op::_rbatcher_id=0; 
int GElib::SO3part_add_CGproduct_back0_op::_rbatcher_id=0; 
int GElib::SO3part_add_CGproduct_back1_op::_rbatcher_id=0; 

int GElib::SO3vec_add_op::_rbatcher_id=0; 
int GElib::SO3vec_add_prod_c_A_op::_rbatcher_id=0; 

int GElib::SO3vec_add_inp_op::_rbatcher_id=0; 

int GElib::SO3vec_add_CGproduct_op::_rbatcher_id=0; 
int GElib::SO3vec_add_CGproduct_back0_op::_rbatcher_id=0; 
int GElib::SO3vec_add_CGproduct_back1_op::_rbatcher_id=0; 

*/


