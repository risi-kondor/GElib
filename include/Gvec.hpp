// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibGvec
#define _GElibGvec

#include "GElib_base.hpp"
#include "SO3partArrayView.hpp"
//#include "TensorPack.hpp"
#include "TensorArrayVirtual.hpp"


namespace GElib{

  template<typename BASE>
  class Gvec: public BASE{
  public:

    using BASE::BASE;
    using BASE::parts;
    

  public: // ---- Copying ------------------------------------------------------------------------------------


    Gvec(const Gvec& x){
      GELIB_COPY_WARNING();
      for(auto& p:x.parts)
	parts[p.first]=static_cast<decltype(BASE::part(0))*>(p.second->clone());
    }
    
    Gvec& operator=(const Gvec& x){
      GELIB_ASSIGN_WARNING();
      for(auto& p:parts)
	delete p->second;
      parts.clear();
      for(auto& p:x.parts)
	parts[p.first]=static_cast<decltype(BASE::part(0))*>(p.second->clone());
    }


  public: // ---- Named constructors -------------------------------------------------------------------------
  
    //static Gvec zero(const TAU& tau, const int _dev=0){
    //Gvec R;
    //return R;
    //}

  };

}

#endif 
