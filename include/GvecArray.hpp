// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibGvecArray
#define _GElibGvecArray

#include "GElib_base.hpp"
#include "SO3partArrayView.hpp"
//#include "TensorPack.hpp"
#include "TensorArrayVirtual.hpp"


namespace GElib{

  template<typename BASE>
  class GvecArray: public BASE{
  public:

    using BASE::BASE;
    using BASE::parts;
    

  public: // ---- Copying ------------------------------------------------------------------------------------


    GvecArray(const GvecArray& x){
      GELIB_COPY_WARNING();
      for(auto& p:x.parts)
	parts[p.first]=static_cast<decltype(BASE::part(0))*>(p.second->clone());
    }
    
    GvecArray& operator=(const GvecArray& x){
      GELIB_ASSIGN_WARNING();
      for(auto& p:parts)
	delete p->second;
      parts.clear();
      for(auto& p:x.parts)
	parts[p.first]=static_cast<decltype(BASE::part(0))*>(p.second->clone());
    }


  
  };

}

#endif 
