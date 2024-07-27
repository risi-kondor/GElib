
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
#include "Ggroup.hpp"
#include "GtypeE.hpp"

namespace GElib{

  template<typename SPEC>
  class GvecSpecBase{
  public:

    typedef cnine::Gdims Gdims;
    typedef cnine::DimLabels DimLabels;

    mutable Gdims adims;
    int nbatch=0;
    int _fcode=0;
    int _dev=0;

    shared_ptr<Ggroup> G;
    shared_ptr<GtypeE> _tau;

    GvecSpecBase(Ggroup* _G): G(_G){
    }

    /*
    GvecSpecBase(const cnine::Gdims& dims, const cnine::DimLabels& labels, const int d, 
      const shared_ptr<Ggroup>& _G, const shared_ptr<GirrepIx> _ix):
      G(_G),
      ix(_ix){
      if(labels._batched) nbatch=dims[0];
      adims=dims.chunk(labels._batched,labels._narray);
      ddims=dims.chunk(labels._batched+labels._narray);
      _dev=d;
    }
    */
      
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

    //SPEC tau(GtypeE* x){_tau.reset(x); return *this;}
    //SPEC tau(const GtypeE& x){_tau.reset(x); return *this;}


  public: // ---- Access ------------------------------------------------------------------------------------


    int get_fcode() const{
      return _fcode;
    }

    int get_dev() const{
      return _dev;
    }

  };


  class GvecSpec: public GvecSpecBase<GvecSpec>{
  public:

    typedef GvecSpecBase<GvecSpec> BASE;
    using BASE::BASE;

    GvecSpec(const BASE& x):
      BASE(x){}


  };

}

#endif 
