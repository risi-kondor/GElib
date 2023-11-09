
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibGvecSpec
#define _GElibGvecSpec

#include "GElib_base.hpp"
#include "LtensorSpec.hpp"
#include "SO3type.hpp"


namespace GElib{


  template<typename GROUP>
  class GvecSpec: public cnine::LtensorSpecBase<typename GROUP::vecSpec>{
  public:

    typedef cnine::LtensorSpecBase<typename GROUP::vecSpec> BASE;
    typedef typename GROUP::vecSpec SPEC;
    typedef typename GROUP::TAU TAU;

    typedef cnine::Gdims Gdims;

    using BASE::BASE;
    using BASE::nbatch;
    using BASE::adims;
    using BASE::_fcode;
    using BASE::_dev;

    TAU _tau;

    GvecSpec(){
    }

    GvecSpec(const BASE& x): 
      BASE(x){}

    GvecSpec(const GvecSpec& x):
      BASE(x),
      _tau(x._tau){}


  public: // ---- Construction ------------------------------------------------------------------------------


    SPEC zero() {_fcode=0; return *this;}
    SPEC raw() {_fcode=1; return *this;}
    SPEC ones() {_fcode=2; return *this;}
    SPEC sequential() {_fcode=3; return *this;}
    SPEC gaussian() {_fcode=4; return *this;}
    SPEC fcode(const int x) {_fcode=x; return *this;}

    SPEC batch(const int b) {nbatch=b; return *this;}

    SPEC grid(const initializer_list<int>& v) {adims=Gdims(v); return *this;}
    SPEC grid(const vector<int>& v) {adims=Gdims(v); return *this;}
    SPEC grid(const Gdims& v) {adims=v; return *this;}

    SPEC tau(const TAU& __tau) {_tau=__tau;return *this;}
    SPEC tau(const initializer_list<int>& list) {_tau=TAU(list); return *this;}

    SPEC fourier(const int maxl) {_tau=TAU::Fourier(maxl);}
    SPEC fourier(const initializer_list<int>& list) {_tau=TAU::Fourier(list); return *this;}

    SPEC dev(const int i) {_dev=i; return *this;}

    


  };

}

#endif 
