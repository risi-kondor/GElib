// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef _SO3groupE
#define _SO3groupE

#include <cnine/tensors>
#include <cnine/containers>

#include "Ggroup.hpp"
#include "GirrepIx.hpp"
#include "GpartSpec.hpp"
#include "GpartE.hpp"

#include "SO3irrepIx.hpp"

#include "SO3part_addCGproductFn.hpp"
#include "SO3part_addCGproduct_back0Fn.hpp"
#include "SO3part_addCGproduct_back1Fn.hpp"
#include "SO3part_addBlockedCGproductFn.hpp"
#include "SO3part_addBlockedCGproduct_back0Fn.hpp"
#include "SO3part_addBlockedCGproduct_back1Fn.hpp"
#include "SO3part_addFproductFn.hpp"
#include "SO3part_addFproduct_back0Fn.hpp"
#include "SO3part_addFproduct_back1Fn.hpp"


namespace GElib{




  class SO3group: public Ggroup{
  public:

    typedef cnine::Ctensor3_view Ctensor3_view;

    GirrepIx* new_irrep(const int x){
      return new SO3irrepIx(x);
    }

    
  public: // ---- CG-products --------------------------------------------------------------------------------


    void addCGproduct(const Ctensor3_view& _r, const Ctensor3_view& _x, const Ctensor3_view& _y, const int _offs=0){
      SO3part_addCGproductFn()(_r,_x,_y,_offs);
    }

    void addCGproduct_back0(const Ctensor3_view& _r, const Ctensor3_view& _g, const Ctensor3_view& _y, const int _offs=0){
      SO3part_addCGproduct_back0Fn()(_r,_g,_y,_offs);
    }

    void addCGproduct_back1(const Ctensor3_view& _r, const Ctensor3_view& _g, const Ctensor3_view& _x, const int _offs=0){
      SO3part_addCGproduct_back1Fn()(_r,_g,_x,_offs);
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "GElib::SO3group";
    }

    string repr() const{
      return "SO(3)";
    }
    
  };

}

#endif 


