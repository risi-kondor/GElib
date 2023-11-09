// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef _HomMapD
#define _HomMapD

#include <cnine/tensors>
#include "LtensorApackSpec.hpp"
#include "LtensorApack.hpp"
#include "GvecD.hpp"

namespace GElib{

  
  template<typename GROUP, typename TYPE>
  class HomMapSpec: public cnine::LtensorApackSpecBase<typename GROUP::IrrepIx,HomMapSpec<GROUP,TYPE> >{
  public:

    typedef cnine::LtensorApackSpecBase<typename GROUP::IrrepIx,HomMapSpec<GROUP,TYPE> > BASE;
    typedef typename GROUP::TAU TAU;

    using BASE::BASE;
    using BASE::nbatch;
    using BASE::ddims;
    using BASE::fcode;
    using BASE::_dev;

    HomMapSpec& dims(const TAU& x, const TAU& y){
      ddims.clear();
      for(auto& p: x)
	ddims.emplace(p.first,cnine::Gdims(p.second,y(p.first)));
      return *this;
    }

    HomMapSpec& gaussian(const TAU& x, const TAU& y) {fcode=4; dims(x,y); return *this;}

  };


  template<typename GROUP, typename TYPE>
  class HomMapD: public cnine::LtensorApack<typename GROUP::IrrepIx,complex<TYPE> >{
  public:

    typedef cnine::LtensorApack<typename GROUP::IrrepIx, complex<TYPE> > BASE;
    typedef cnine::Ltensor<TYPE> TENSOR;
    typedef typename GROUP::IrrepIx _IrrepIx;

    using BASE::BASE;
    using BASE::_nbatch;
    using BASE::_gdims;
    using BASE::_dev;

    ~HomMapD(){
    }


  public: // ---- Constructors ------------------------------------------------------------------------------


    HomMapD(){}

    HomMapD(const HomMapSpec<GROUP,TYPE>& spec){
      _nbatch=spec.nbatch;
      _gdims=spec.gdims;
      _dev=spec._dev;
    }

  };




  template<typename GROUP, typename TYPE>
  inline GvecD<GROUP,TYPE> operator*(const GvecD<GROUP,TYPE>& x, const HomMapD<GROUP,TYPE>& y){
    GvecD<GROUP,TYPE> r(x._nbatch,x._gdims,x.dev);
    typedef decltype(GROUP::template dummy_part<TYPE>()) PART;
    for(auto& p: x.parts)
      r.parts[p.first]=new PART((*p.second)*y[p.first]);
    //r.parts.emplace(p.first,(*p.second)*y[p.first]);
    return r;
  }

}

#endif 
