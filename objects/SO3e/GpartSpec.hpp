
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibGpartSpec
#define _GElibGpartSpec

#include "Ltensor.hpp"
#include "GElib_base.hpp"
#include "Ggroup.hpp"


namespace GElib{


  template<typename SPEC>
  class GpartSpecBase{
  public:

    typedef cnine::Gdims Gdims;
    typedef cnine::DimLabels DimLabels;

    mutable Gdims adims;
    Gdims ddims;
    int nbatch=0;
    int _fcode=0;
    int _dev=0;

    shared_ptr<Ggroup> G;
    shared_ptr<GirrepIx> ix;

    GpartSpecBase(Ggroup* _G): G(_G){
      ddims=cnine::Gdims(0,0);
    }

    GpartSpecBase(const cnine::Gdims& dims, const cnine::DimLabels& labels, const int d, 
      const shared_ptr<Ggroup>& _G, const shared_ptr<GirrepIx> _ix):
      G(_G),
      ix(_ix){
      if(labels._batched) nbatch=dims[0];
      adims=dims.chunk(labels._batched,labels._narray);
      ddims=dims.chunk(labels._batched+labels._narray);
      _dev=d;
    }

      
  public: // ---- Copying -------------------------------------------------------------------------------------



  public: // ---- Construction --------------------------------------------------------------------------------


    SPEC batch(const int b) {nbatch=b; return *this;}
    SPEC batches(const int b) {nbatch=b; return *this;}

    SPEC grid(const initializer_list<int>& v) {adims=Gdims(v); return *this;}
    SPEC grid(const vector<int>& v) {adims=Gdims(v); return *this;}
    SPEC grid(const Gdims& v) {adims=v; return *this;}

    SPEC zero() {_fcode=0; return *this;}
    SPEC raw() {_fcode=1; return *this;}
    SPEC ones() {_fcode=2; return *this;}
    SPEC sequential() {_fcode=3; return *this;}
    SPEC gaussian() {_fcode=4; return *this;}
    SPEC fcode(const int x) {_fcode=x; return *this;}

    SPEC dev(const int i) {_dev=i; return *this;}

    SPEC irrep(GirrepIx* x){ix.reset(x); return *this;}

    SPEC n(const int nc){
      GELIB_ASSRT(ddims.size()==2);
      ddims[1]=nc;
      return *this;
    }


  public: // ---- Access ------------------------------------------------------------------------------------


    Gdims get_dims() const{
      Gdims r;
      if(nbatch>0) r.extend({nbatch});
      if(adims.size()!=0) r.extend(adims);
      if(ddims.size()!=0) r.extend(ddims);
      return r;
    }

    DimLabels get_labels() const{
      DimLabels r;
      if(nbatch>0) r._batched=true;
      r._narray=adims.size();
      return r;
    }

    int get_fcode() const{
      return _fcode;
    }

    int get_dev() const{
      return _dev;
    }

  };


  class GpartSpec: public GpartSpecBase<GpartSpec>{
  public:

    typedef GpartSpecBase<GpartSpec> BASE;
    using BASE::BASE;

    GpartSpec(const BASE& x):
      BASE(x){}


  };


}

#endif 
