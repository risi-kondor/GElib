// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibBatchedSO3vecView
#define _GElibBatchedSO3vecView

#include "GElib_base.hpp"
#include "BatchedGvecView.hpp"
#include "BatchedSO3partView.hpp"
#include "SO3vecView.hpp"
#include "BatchedSO3vecView.hpp"


namespace GElib{

  template<typename RTYPE>
  class BatchedSO3vecView: public BatchedGvecView<int,BatchedSO3partView<RTYPE>,SO3vecView<RTYPE> >{
  public:

    typedef BatchedGvecView<int,BatchedSO3partView<RTYPE>,SO3vecView<RTYPE> > BatchedGvecView;
    typedef SO3partView<RTYPE> SO3partView;

    using BatchedGvecView::BatchedGvecView;
    using BatchedGvecView::parts;


  public: // ---- Access ------------------------------------------------------------------------------------


    int get_maxl() const{
      int r=0;
      for(auto& p:parts)
	r=std::max(r,p.first);
      return r;
    }
    
    SO3type get_tau() const{
      SO3type tau(parts.size(),cnine::fill_raw());
      for(auto& p:parts)
	tau[p.first]=p.second->getn();
      return tau;
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    static string classname(){
      return "GElib::BatchedSO3vecView";
    }

    /*
    string str(const string indent="") const{
      ostringstream oss;
      for_each_batch([&](const int b, const VEC& x){
	  oss<<indent<<"Batch "<<b<<":"<<endl;
	  oss<<indent<<x<<endl;
	});
      //for(int l=0; l<parts.size(); l++){
	//if(!parts[l]) continue;
      //oss<<indent<<"Part l="<<l<<":\n";
      //oss<<(*this)(l).str(indent+"  ");
      //oss<<endl;
      //}
      return oss.str();
    }
    */

    string repr(const string indent="") const{
      return "";
      //return "<GElib::SO3vecV of type "+get_tau().str()+">";
    }
    
    friend ostream& operator<<(ostream& stream, const BatchedSO3vecView& x){
      stream<<x.str(); return stream;
    }

    

  };


}

#endif 
