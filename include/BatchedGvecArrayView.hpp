// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibBatchedGvecArrayView
#define _GElibBatchedGvecArrayView

#include "GElib_base.hpp"
#include "GvecArrayView.hpp"


namespace GElib{

  template<typename KEY, typename BPART, typename VEC_VIEW>
  class BatchedGvecArrayView: public GvecArrayView<KEY,BPART,VEC_VIEW>{
  public:

    typedef GvecArrayView<KEY,BPART,VEC_VIEW> GvecArrayView;

    using GvecArrayView::GvecArrayView;
    using GvecArrayView::parts;


  public: // ---- Copying -----------------------------------------------------------------------------------


  public: // ---- Access ------------------------------------------------------------------------------------


    int getb() const{
      return parts.begin()->second->getb();
    }

    VEC_VIEW batch(const int b) const{
      CNINE_CHECK_RANGE(b<getb());
      VEC_VIEW R;
      for(auto& p:parts)
	R.parts[p.first]=p.second->batch(b).clone();
      return R;
    }



  public: // ---- Lambdas ------------------------------------------------------------------------------------


    //void for_each_part(const std::function<void(const KEY&, const PART&)>& lambda) const{
    //for(auto& p:parts) 
    //lambda(p.first,*p.second);
    //}

    void for_each_batch(const std::function<void(const int, const VEC_VIEW& x)>& lambda) const{
      int B=getb();
      for(int b=0; b<B; b++)
	lambda(b,batch(b));
    }
    

  public: // ---- Cumulative operations ----------------------------------------------------------------------


    void add(const BatchedGvecArrayView& x){
      for(auto p: parts){
	p.second->add(x.part(p.first));
      }
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      for_each_batch([&](const int b, const VEC_VIEW& x){
	  oss<<indent<<"Batch "<<b<<":"<<endl;
	  oss<<indent<<x<<endl;
	});
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const BatchedGvecArrayView& x){
      stream<<x.str(); return stream;
    }


  };

}

#endif 
    /*
    BatchedGvecArrayView(const BatchedGvecArrayView& x){
      GELIB_COPY_WARNING();
      for(auto& p:x.parts)
	parts[p.first]=new PART(*p.second);
    }
    
    BatchedGvecArrayView(BatchedGvecArrayView&& x):
      parts(std::move(x.parts)){
      GELIB_MOVE_WARNING();
    }
      
    BatchedGvecArrayView& operator=(const BatchedGvecArrayView& x){
      GELIB_ASSIGN_WARNING();
      for(auto& p:parts)
	(*p.second)=(*x.parts[p.first]);
      return *this;
    }
    */


