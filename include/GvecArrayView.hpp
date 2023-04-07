// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibGvecArrayView
#define _GElibGvecArrayView

#include "GElib_base.hpp"


namespace GElib{

  template<typename KEY, typename PART, typename CELL>
  class GvecArrayView{
  public:

    typedef cnine::Gdims Gdims;
    typedef cnine::Gindex Gindex;

    mutable unordered_map<KEY,PART*> parts;

    GvecArrayView(){}

    ~GvecArrayView(){
      for(auto& p: parts)
	delete p.second;
    }


  public: // ---- Copying -----------------------------------------------------------------------------------


    GvecArrayView(const GvecArrayView& x){
      GELIB_COPY_WARNING();
      for(auto& p:x.parts)
	parts[p.first]=new PART(*p.second);
    }
    
    GvecArrayView(GvecArrayView&& x):
      parts(std::move(x.parts)){
      GELIB_MOVE_WARNING();
    }
      
    GvecArrayView& operator=(const GvecArrayView& x){
      GELIB_ASSIGN_WARNING();
      for(auto& p:parts)
	(*p.second)=(*x.parts[p.first]);
    }


  public: // ---- Access ------------------------------------------------------------------------------------


    int nadims() const{
      GELIB_ASSRT(parts.size()>0);
      return parts.begin()->ak;
    }

    Gdims get_adims() const{
      GELIB_ASSRT(parts.size()>0);
      return parts.begin()->get_adims();
    }


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

    CELL cell(const int i0) const{
      CELL R;
      for(auto& p:parts)
	R.parts[p.first]=new decltype(R.part(0))((*p.second)(i0));
      return R;
    }

    CELL cell(const int i0, const int i1) const{
      CELL R;
      for(auto& p:parts)
	R.parts[p.first]=new decltype(R.part(0))((*p.second)(i0,i1));
      return R;
    }

    CELL cell(const int i0, const int i1, const int i2) const{
      CELL R;
      for(auto& p:parts)
	R.parts[p.first]=new decltype(R.part(0))((*p.second)(i0,i1,i2));
      return R;
    }

    CELL cell(const Gindex& ix) const{
      CELL R;
      for(auto& p:parts)
	R.parts[p.first]=new decltype(R.part(0))((*p.second)(ix));
      return R;
    }


  public: // ---- Lambdas ------------------------------------------------------------------------------------


    void for_each_part(const std::function<void(const KEY&, const PART&)>& lambda) const{
      for(auto& p:parts) 
	lambda(p.first,*p.second);
    }

    void for_each_cell(const std::function<void(const Gindex&, const CELL&)>& lambda) const{
      get_adims().for_each_index([&](const Gindex& ix){
	  lambda(ix,cell(ix));});
    }


  public: // ---- Cumulative operations ----------------------------------------------------------------------


    void add(const GvecArrayView& x){
      for(auto p: parts){
	p.second->add(x.part(p.first));
      }
    }

    void add(const CELL& x){
      for(auto p: parts){
	p.second->add(x.part(p.first));
      }
    }

    



  public:

    

  };


}

#endif 
