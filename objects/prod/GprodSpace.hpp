// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef _GprodSpace
#define _GprodSpace


namespace GElib{


  template<typename GROUP>
  class GprodSpace{
  public:

    typedef typename GROUP::IrrepIx _IrrepIx;
    typedef GprodSpaceObj<GROUP> OBJ;


    OBJ* obj;

    GprodSpace(OBJ* _obj):
      obj(_obj){}

    GprodSpace(const _IrrepIx& ix):
      GprodSpace(GROUP::space(ix)){}



  public: // ---- Access ------------------------------------------------------------------------------------


    bool is_leaf() const{
      return obj->is_leaf();
    }

    bool is_isomorphic(const GprodSpace& y) const{
      return obj->is_isomorphic(*y.obj);
    }

    GprodSpace left() const{
      GELIB_ASSRT(!is_leaf());
      return obj->left;
    }

    GprodSpace right() const{
      GELIB_ASSRT(!is_leaf());
      return obj->right;
    }

    Gtype<GROUP> get_tau() const{
      return obj->get_tau();
    }


  public: // ---- Operations --------------------------------------------------------------------------------


    GprodSpace FmoveL() const{
      return GROUP::FmoveL(obj);
    }

    GprodSpace FmoveR() const{
      return GROUP::FmoveR(obj);
    }

    /*
    void canonicalize() const{
      if(is_leaf()) return;
      GprodSpace R=right();
      R.canonicalizeR();
      while(!R.is_leaf()){
      }
    }
    */


    /*
    HomMap<GROUP,double> coupling(const GprodSpace& y){
      return GROUP::coupling(obj,y.obj);
    }
    */

  public: // ---- I/O ---------------------------------------------------------------------------------------


    string repr() const{
      return obj->repr();
    }

    string str(const string indent="") const{
      return obj->str(indent);
    }

    friend ostream& operator<<(ostream& stream, const GprodSpace& x){
      stream<<x.str(); return stream;
    }


  };


  template<typename GROUP>
  inline GprodSpace<GROUP> operator*(const GprodSpace<GROUP>& x, const GprodSpace<GROUP>& y){
    return GprodSpace(GROUP::space(x.obj,y.obj));
  }

}

#endif 
