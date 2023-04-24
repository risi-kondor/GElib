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

  template<typename KEY, typename Pview, typename Vview>
  class GvecView{
  public:

    mutable map<KEY,Pview*> parts;

    GvecView(){}

    ~GvecView(){
      for(auto& p: parts)
	delete p.second;
    }


  public: // ---- Copying -----------------------------------------------------------------------------------


    GvecView(const GvecView& x){
      GELIB_COPY_WARNING();
      for(auto& p:x.parts)
	parts[p.first]=new Pview(*p.second);
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


  public: // ---- ATen --------------------------------------------------------------------------------------

    
    #ifdef _WITH_ATEN
    
    vector<at::Tensor> torch() const{
      vector<at::Tensor> R;
      for_each_part([&](const KEY& key, const Pview& part){
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

    Vview batch(const int b) const{
      CNINE_CHECK_RANGE(b<getb());
      Vview R;
      for(auto& p:parts)
	R.parts[p.first]=new Pview(p.second->batch(b));
      //R.parts[p.first]=p.second->batch(b).clone();
      return R;
    }

    int device() const{
      if(parts.size()==0) return 0;
      return parts.begin()->second->device();
    }

    Pview operator()(const KEY& l) const{
      auto it=parts.find(l);
      assert(it!=parts.end());
      return Pview(*it->second);
    }

    Pview part(const KEY& l) const{
      auto it=parts.find(l);
      assert(it!=parts.end());
      return Pview(*it->second);
    }


  public: // ---- Lambdas ------------------------------------------------------------------------------------


    void for_each_part(const std::function<void(const KEY&, const Pview&)>& lambda) const{
      for(auto& p:parts) 
	lambda(p.first,*p.second);
    }

    void for_each_batch(const std::function<void(const int, const Vview& x)>& lambda) const{
      int B=getb();
      for(int b=0; b<B; b++)
	lambda(b,batch(b));
    }

  public: // ---- Cumulative operations ----------------------------------------------------------------------


    void add(const GvecView& x){
      for(auto p: parts){
	p.second->add(x.part(p.first));
      }
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      if(getb()>1){
	for_each_batch([&](const int b, const GvecView& x){
	    oss<<indent<<"Batch "<<b<<":"<<endl;
	    oss<<x.str(indent+"  ");
	  });
      }else{
	for_each_part([&](const int p, const Pview& x){
	    oss<<indent<<"Part "<<p<<":"<<endl;
	    oss<<x.str(indent);
	  });
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const GvecView& x){
      stream<<x.str(); return stream;
    }


  };

  

  
}

#endif 
