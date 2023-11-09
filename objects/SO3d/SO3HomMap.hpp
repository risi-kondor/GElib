// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibSO3HomMap
#define _GElibSO3HomMap

#include "GElib_base.hpp"
#include "diff_class.hpp"
#include "SO3group.hpp"
#include "HomMapD.hpp"


namespace GElib{

  template<typename TYPE>
  class SO3HomMap: public HomMapD<SO3group,TYPE>{
  public:

    typedef HomMapD<SO3group,TYPE> BASE;

    using BASE::BASE;
    using BASE::tensors;


  public: // ---- HomMapSpec<SO3group> -------------------------------------------------------------------------------


    SO3HomMap(const HomMapSpec<SO3group,TYPE>& spec):
      BASE(spec){
      cnine::DimLabels labels=spec.get_labels(); 
      if(spec.nbatch>0)
	for(auto p: spec.ddims)
	  tensors.emplace(p.first,cnine::Ltensor<complex<TYPE> >(spec.gdims.cat(p.second).prepend(spec.nbatch),
	      labels,spec.fcode,spec._dev));
      else
	for(auto p: spec.ddims)
	  tensors.emplace(p.first,cnine::Ltensor<complex<TYPE> >(spec.gdims.cat(p.second),
	      labels,spec.fcode,spec._dev));
    }
    
    static HomMapSpec<SO3group,TYPE> raw() {return HomMapSpec<SO3group,TYPE>().raw();}
    static HomMapSpec<SO3group,TYPE> zero() {return HomMapSpec<SO3group,TYPE>().zero();}
    static HomMapSpec<SO3group,TYPE> sequential() {return HomMapSpec<SO3group,TYPE>().sequential();}
    static HomMapSpec<SO3group,TYPE> gaussian() {return HomMapSpec<SO3group,TYPE>().gaussian();}

    static HomMapSpec<SO3group,TYPE> gaussian(const SO3typeD& x, const SO3typeD& y){
      return HomMapSpec<SO3group,TYPE>().gaussian(x,y);}

    HomMapSpec<SO3group,TYPE> spec() const{
      return BASE::spec();
    }



  };

}

#endif 
