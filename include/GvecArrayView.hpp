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

  // make this a subclass of GvecView? 
  template<typename KEY, typename PAview, typename VAview, typename Vview>
  class GvecArrayView{
  public:

    typedef cnine::Gdims Gdims;
    typedef cnine::Gindex Gindex;

    mutable unordered_map<KEY,PAview*> parts;

    GvecArrayView(){}

    ~GvecArrayView(){
      for(auto& p: parts)
	delete p.second;
    }


  public: // ---- Copying -----------------------------------------------------------------------------------


    GvecArrayView(const GvecArrayView& x){
      GELIB_COPY_WARNING();
      for(auto& p:x.parts)
	parts[p.first]=new PAview(*p.second);
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


  public: // ---- ATen --------------------------------------------------------------------------------------

    
    #ifdef _WITH_ATEN
    
    vector<at::Tensor> torch() const{
      vector<at::Tensor> R;
      for_each_part([&](const KEY& key, const PAview& part){
	  R.push_back(part.torch());});
      return R;
    }

    #endif 


  public: // ---- Access ------------------------------------------------------------------------------------

    
    int size() const{
      return parts.size();
    }

    int getb() const{
      return parts.begin()->second->getb();
    }
    
    int device() const{
      if(parts.size()==0) return 0;
      return parts.begin()->second->device();
    }

    int nadims() const{
      GELIB_ASSRT(parts.size()>0);
      return parts.begin()->second->ak;
    }

    Gdims get_adims() const{
      GELIB_ASSRT(parts.size()>0);
      return parts.begin()->second->get_adims();
    }


    VAview batch(const int b) const{
      //CNINE_CHECK_RANGE(b<getb());
      VAview R;
      for(auto& p:parts)
	R.parts[p.first]=new PAview(p.second->batch(b));
      return R;
    }

    PAview operator()(const KEY& l) const{
      auto it=parts.find(l);
      assert(it!=parts.end());
      return PAview(*it->second);
    }

    PAview part(const KEY& l) const{
      auto it=parts.find(l);
      assert(it!=parts.end());
      return PAview(*it->second);
    }

    Vview cell(const int i0) const{
      Vview R;
      for(auto& p:parts)
	R.parts[p.first]=new decltype(R.part(0))((*p.second)(i0));
      return R;
    }

    Vview cell(const int i0, const int i1) const{
      Vview R;
      for(auto& p:parts)
	R.parts[p.first]=new decltype(R.part(0))((*p.second)(i0,i1));
      return R;
    }

    Vview cell(const int i0, const int i1, const int i2) const{
      Vview R;
      for(auto& p:parts)
	R.parts[p.first]=new decltype(R.part(0))((*p.second)(i0,i1,i2));
      return R;
    }

    Vview cell(const Gindex& ix) const{
      Vview R;
      for(auto& p:parts)
	R.parts[p.first]=new decltype(R.part(0))((*p.second)(ix));
      return R;
    }


  public: // ---- Lambdas ------------------------------------------------------------------------------------


    void for_each_batch(const std::function<void(const int, const VAview& x)>& lambda) const{
      int B=getb();
      for(int b=0; b<B; b++)
	lambda(b,batch(b));
    }

    void for_each_part(const std::function<void(const KEY&, const PAview&)>& lambda) const{
      for(auto& p:parts) 
	lambda(p.first,*p.second);
    }

    void for_each_cell(const std::function<void(const Gindex&, const Vview&)>& lambda) const{
      get_adims().for_each_index([&](const Gindex& ix){
	  lambda(ix,cell(ix));});
    }


  public: // ---- Cumulative operations ----------------------------------------------------------------------


    void add(const GvecArrayView& x){
      for(auto p: parts){
	p.second->add(x.part(p.first));
      }
    }

    //void add(const Vview& x){
    //for(auto p: parts){
    //p.second->add(x.part(p.first));
    //}
    //}

    
  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      if(getb()>1){
	for_each_batch([&](const int b, const GvecArrayView& x){
	    oss<<indent<<"Batch "<<b<<":"<<endl;
	    oss<<x.str(indent+"  ");
	  });
      }else{
	for_each_part([&](const KEY p, const PAview& x){
	    oss<<indent<<"Part "<<p<<":"<<endl;
	    oss<<x.str(indent+"  ");
	  });
      }
      return oss.str();
    }




  public:

    

  };


}

#endif 
