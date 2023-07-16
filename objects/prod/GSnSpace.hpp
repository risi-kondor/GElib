// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef _GSnSpace
#define _GSnSpace

#include "GSnSpaceBank.hpp"


namespace GElib{



  template<typename GROUP>
  class GSnSpace{
  public:

    typedef typename GROUP::IrrepIx _IrrepIx;
    typedef GSnSpaceObj<GROUP> OBJ;


    OBJ* obj;

    GSnSpace(OBJ* _obj):
      obj(_obj){}

    GSnSpace(const _IrrepIx& ix):
      GSnSpace(GSnSpace_bank(ix)){}



  public: // ---- Operations --------------------------------------------------------------------------------


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string repr() const{
      return obj->repr();
    }

    string str(const string indent="") const{
      return "";
    }

    friend ostream& operator<<(ostream& stream, const GSnSpace& x){
      stream<<x.str(); return stream;
    }


  };


  template<typename GROUP>
  inline GSnSpace<GROUP> operator*(const GSnSpace<GROUP>& x, const GSnSpace<GROUP>& y){
    return GSnSpace(GSnSpace_bank(x.obj,y.obj));
  }

}

#endif 
