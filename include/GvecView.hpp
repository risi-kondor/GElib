// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibGvecView
#define _GElibGvecView

#include "GElib_base.hpp"


namespace GElib{

  template<typename KEY, typename PART>
  class GvecView{
  public:

    mutable unordered_map<KEY,PART*> parts;

    GvecView(){}

    ~GvecView(){
      for(auto& p: parts)
	delete p.second;
    }


  public: // ---- Copying -----------------------------------------------------------------------------------


    GvecView(const GvecView& x){
      GELIB_COPY_WARNING();
      for(auto& p:x.parts)
	parts[p.first]=new PART(*p.second);
    }
    
    GvecView(GvecView&& x):
      parts(std::move(x.parts)){
      GELIB_MOVE_WARNING();
    }
      
    GvecView& operator=(const GvecView& x){
      GELIB_ASSIGN_WARNING();
      for(auto& p:parts)
	(*p.second)=(*x.parts[p.first]);
      return *this;
    }


  public: // ---- Access ------------------------------------------------------------------------------------


    PART operator()(const KEY& l) const{
      auto it=parts.find(l);
      assert(it!=parts.end());
      return PART(*it->second);
    }

    PART part(const KEY& l) const{
      auto it=parts.find(l);
      assert(it!=parts.end());
      return PART(*it->second);
    }


  public: // ---- Lambdas ------------------------------------------------------------------------------------


    void for_each_part(const std::function<void(const KEY&, const PART&)>& lambda) const{
      for(auto& p:parts) 
	lambda(p.first,*p.second);
    }


  public: // ---- Cumulative operations ----------------------------------------------------------------------


    void add(const GvecView& x){
      for(auto p: parts){
	p.second->add(x.part(p.first));
      }
    }

  };

  
}

#endif 
