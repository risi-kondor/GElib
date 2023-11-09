
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibSO3partSpec
#define _GElibSO3partSpec

#include "GElib_base.hpp"
#include "LtensorSpec.hpp"


namespace GElib{

  //template<typename TYPE>
  //class SO3part;

  //template<typename TYPE>
  class SO3partSpec: public cnine::LtensorSpecBase<SO3partSpec>{
  public:

    typedef cnine::LtensorSpecBase<SO3partSpec> BASE;
    using BASE::BASE;

    using BASE::adims;
    using BASE::ddims;
    using BASE::nbatch;
    using BASE::_fcode;
    using BASE::_dev;


    SO3partSpec(){
      ddims=cnine::Gdims(0,0);
    }

    SO3partSpec(const BASE& x): 
      BASE(x){
      if(ddims.size()!=2) ddims=cnine::Gdims(0,0);
    }

    
    template<typename SPEC>
    SO3partSpec(const cnine::LtensorSpecBase<SPEC>& x){
      adims=x.adims;
      ddims=cnine::Gdims(0,0);
      nbatch=x.nbatch;
      _fcode=x._fcode;
      _dev=x._dev;
    }
    
      
    template<typename TYPE>
    SO3partSpec(const cnine::LtensorSpec<complex<TYPE> > x): 
      BASE(reinterpret_cast<const BASE&>(x)){
      if(ddims.size()!=2) ddims=cnine::Gdims(0,0);
    }

    //SO3part<TYPE> operator()(){
    //return SO3part<TYPE>(*this);
    //}
    

  public: // ---- Copying -----------------------------------



  public: // ---- Construction ------------------------------


    SO3partSpec l(const int _l){
      GELIB_ASSRT(ddims.size()==2);
      ddims[0]=2*_l+1;
      return *this;
    }

    SO3partSpec n(const int nc){
      GELIB_ASSRT(ddims.size()==2);
      ddims[1]=nc;
      return *this;
    }

    SO3partSpec channels(const int nc){
      ddims[1]=nc;
      return *this;
    }

  };


}

#endif 
