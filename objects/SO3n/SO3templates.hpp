
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibSO3templates
#define _GElibSO3templates

#include "GElib_base.hpp"

#include "SO3part_addCGproductFn.hpp"
#include "SO3part_addCGproduct_back0Fn.hpp"
#include "SO3part_addCGproduct_back1Fn.hpp"


namespace GElib{

  template<typename TYPE0, typename TYPE1>
  void add_CGproduct(const TYPE0& r, const TYPE1& x, const TYPE1& y, const int offs=0){
    SO3part_addCGproductFn()(r,x,y,offs);
  }

  template<typename TYPE0, typename TYPE1>
  void add_CGproduct_back0(const TYPE0& r, const TYPE1& x, const TYPE1& y, const int offs=0){
    SO3part_addCGproduct_back0Fn()(r,x,y,offs);
  }

  template<typename TYPE0, typename TYPE1>
  void add_CGproduct_back1(const TYPE0& r, const TYPE1& x, const TYPE1& y, const int offs=0){
    SO3part_addCGproduct_back1Fn()(r,x,y,offs);
  }


  template<typename TYPE0, typename TYPE1>
  void add_DiagCGproduct(const TYPE0& r, const TYPE1& x, const TYPE1& y, const int offs=0){
    SO3part_addBlockedCGproductFn()(r,x,y,1,offs);
  }

  template<typename TYPE0, typename TYPE1>
  void add_DiagCGproduct_back0(const TYPE0& r, const TYPE1& x, const TYPE1& y, const int offs=0){
    SO3part_addBlockedCGproduct_back0Fn()(r,x,y,1,offs);
  }

  template<typename TYPE0, typename TYPE1>
  void add_DiagCGproduct_back1(const TYPE0& r, const TYPE1& x, const TYPE1& y, const int offs=0){
    SO3part_addBlockedCGproduct_back1Fn()(r,x,y,1,offs);
  }



}


#endif 
