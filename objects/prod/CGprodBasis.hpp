// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef _CGprodBasis
#define _CGprodBasis

#include "CGprodBasisObj.hpp"


namespace GElib{


  template<typename GROUP>
  class CGprodBasis{
  public:

    typedef typename GROUP::IrrepIx _IrrepIx;
    typedef CGprodBasisObj<GROUP> OBJ;


    OBJ* obj;

    CGprodBasis(OBJ* _obj):
      obj(_obj){}

    CGprodBasis(OBJ& _obj):
      obj(&_obj){}

    CGprodBasis(const _IrrepIx& ix):
      CGprodBasis(GROUP::space(ix)){}



  public: // ---- Access ------------------------------------------------------------------------------------


    bool is_leaf() const{
      return obj->is_leaf();
    }

    bool is_isomorphic(const CGprodBasis& y) const{
      return obj->is_isomorphic(*y.obj);
    }

    CGprodBasis left() const{
      GELIB_ASSRT(!is_leaf());
      return obj->left;
    }

    CGprodBasis right() const{
      GELIB_ASSRT(!is_leaf());
      return obj->right;
    }

    Gtype<GROUP> get_tau() const{
      return obj->get_tau();
    }


  public: // ---- Transformations to other bases ------------------------------------------------------------


    CGprodBasis shift_left() const{
      return obj->shift_left();
    }

    CGprodBasis shift_right() const{
      return obj->shift_right();
    }

    CGprodBasis standard_form() const{
      return obj->standard_form();
    }

    CGprodBasis reverse_standard_form() const{
      return obj->reverse_standard_form();
    }

    const EndMap<GROUP,double>& standardizing_map() const{
      return obj->standardizing_map();
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string repr() const{
      return obj->repr();
    }

    string str(const string indent="") const{
      return obj->str(indent);
    }

    friend ostream& operator<<(ostream& stream, const CGprodBasis& x){
      stream<<x.str(); return stream;
    }


  };


  template<typename GROUP>
  inline CGprodBasis<GROUP> operator*(const CGprodBasis<GROUP>& x, const CGprodBasis<GROUP>& y){
    return CGprodBasis(GROUP::space(x.obj,y.obj));
  }

}

#endif 
