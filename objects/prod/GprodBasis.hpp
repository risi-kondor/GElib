// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef _GprodBasis
#define _GprodBasis

#include "GprodSpaceObj.hpp"


namespace GElib{


  template<typename GROUP>
  class GprodBasis{
  public:

    typedef typename GROUP::IrrepIx _IrrepIx;
    typedef GprodSpaceObj<GROUP> OBJ;


    OBJ* obj;

    GprodBasis(OBJ* _obj):
      obj(_obj){}

    GprodBasis(OBJ& _obj):
      obj(&_obj){}

    GprodBasis(const _IrrepIx& ix):
      GprodBasis(GROUP::space(ix)){}



  public: // ---- Access ------------------------------------------------------------------------------------


    bool is_leaf() const{
      return obj->is_leaf();
    }

    bool is_isomorphic(const GprodBasis& y) const{
      return obj->is_isomorphic(*y.obj);
    }

    GprodBasis left() const{
      GELIB_ASSRT(!is_leaf());
      return obj->left;
    }

    GprodBasis right() const{
      GELIB_ASSRT(!is_leaf());
      return obj->right;
    }

    Gtype<GROUP> get_tau() const{
      return obj->get_tau();
    }


  public: // ---- Transformations to other bases ------------------------------------------------------------


    GprodBasis shift_left() const{
      return obj->shift_left();
    }

    GprodBasis shift_right() const{
      return obj->shift_right();
    }

    GprodBasis standard_form() const{
      return obj->standard_form();
    }

    GprodBasis reverse_standard_form() const{
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

    friend ostream& operator<<(ostream& stream, const GprodBasis& x){
      stream<<x.str(); return stream;
    }


  };


  template<typename GROUP>
  inline GprodBasis<GROUP> operator*(const GprodBasis<GROUP>& x, const GprodBasis<GROUP>& y){
    return GprodBasis(GROUP::space(x.obj,y.obj));
  }

}

#endif 
