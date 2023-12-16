
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibGgroup
#define _GElibGgroup

#include "GElib_base.hpp"
#include "GirrepIx.hpp"


namespace GElib{


  class Ggroup{
  public:

    typedef cnine::Ctensor3_view Ctensor3_view;


    virtual ~Ggroup(){}

    virtual GirrepIx* new_irrep(const int x){
      return nullptr;
    }


  public: // ---- I/O --------------------------------------------------------------------------------------


    virtual string repr() const=0;


  public: // ---- CG-products --------------------------------------------------------------------------------


    virtual void addCGproduct(const Ctensor3_view& _r, const Ctensor3_view& _x, const Ctensor3_view& _y, const int _offs=0){
    }

    virtual void addCGproduct_back0(const Ctensor3_view& _r, const Ctensor3_view& _g, const Ctensor3_view& _y, const int _offs=0){
    }

    virtual void addCGproduct_back1(const Ctensor3_view& _r, const Ctensor3_view& _g, const Ctensor3_view& _x, const int _offs=0){
    }

 
  };



}

#endif 


     /*
    template<typename TYPE>
    virtual void add_CGproduct(GpartE<TYPE>& r, const GpartE<TYPE>& x, const GpartE<TYPE>& y, const int _offs=0){
      cout<<333<<endl;
      //G->add_CGproduct(*this,x,y,_offs);
    }

    template<typename TYPE>
    virtual void add_CGproduct_back0(GpartE<TYPE>& r, const GpartE<TYPE>& g, const GpartE<TYPE>& y, const int _offs=0){
      //G->add_CGproduct_back0(*this,g,y,_offs);
    }

    template<typename TYPE>
    virtual void add_CGproduct_back1(GpartE<TYPE>& r, const GpartE<TYPE>& g, const GpartE<TYPE>& x, const int _offs=0){
      //G->add_CGproduct_back1(*this,g,g,_offs);
    }
      */
