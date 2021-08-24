
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _ArithmeticExpr
#define _ArithmeticExpr

namespace GElib{


  template<typename ROBJ, typename XOBJ, typename YOBJ, typename ACT>
  class ArithmeticBinaryExpr{
  public:

    const XOBJ& x;
    const YOBJ& y;

    ArithmeticBinaryExpr(const XOBJ& _x, const YOBJ& _y): x(_x), y(_y){}

    operator ROBJ() const{
      return ACT()(x,y);
    }

  public:

    ROBJ operator*(float c) const{
      ROBJ z=ACT()(x,y);
      ROBJ R(z,cnine::fill::zero);
      R.add(z,c);
      return R;
    }

    ROBJ operator*(double c) const{
      ROBJ z=ACT()(x,y);
      ROBJ R(z,cnine::fill::zero);
      R.add(z,c);
      return R;
    }

    ROBJ operator*(complex<float> c) const{
      ROBJ z=ACT()(x,y);
      ROBJ R(z,cnine::fill::zero);
      R.add(z,c);
      return R;
    }


  public:

    string str() const{
      return ROBJ(*this).str();
    }

    friend ostream& operator<<(ostream& stream, const ArithmeticBinaryExpr& x){
      stream<<x.str(); return stream;
    }

  };


  template<typename XOBJ, typename YOBJ, typename ROBJ, typename P1, typename ACT>
  class ArithmeticBinaryExpr1{
  public:

    const P1 p1;

    const XOBJ& x;
    const YOBJ& y;

    ArithmeticBinaryExpr1(const XOBJ& _x, const YOBJ& _y): x(_x), y(_y){}

    ArithmeticBinaryExpr1(const XOBJ& _x, const YOBJ& _y, const P1& _p1): x(_x), y(_y), p1(_p1){}

    operator ROBJ() const{
      return ACT()(x,y,p1);
    }


  public:

    ROBJ operator*(float c) const{
      ROBJ z=ACT()(x,y,p1);
      ROBJ R(z,cnine::fill::zero);
      R.add(z,c);
      return R;
    }

    ROBJ operator*(double c) const{
      ROBJ z=ACT()(x,y,p1);
      ROBJ R(z,cnine::fill::zero);
      R.add(z,c);
      return R;
    }

    ROBJ operator*(complex<float> c) const{
      ROBJ z=ACT()(x,y,p1);
      ROBJ R(z,cnine::fill::zero);
      R.add(z,c);
      return R;
    }


  public:

    string str() const{
      return ROBJ(*this).str();
    }

    friend ostream& operator<<(ostream& stream, const ArithmeticBinaryExpr1& x){
      stream<<x.str(); return stream;
    }

  };


}

#endif 
