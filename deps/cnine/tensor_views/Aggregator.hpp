/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2021, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#ifndef _Aggregator
#define _Aggregator

#include "Cnine_base.hpp"
#include "Rmask1.hpp"
#include "Ctensor2_view.hpp"
#include "Ctensor3_view.hpp"
#include "Ctensor4_view.hpp"
#include "CtensorD_view.hpp"
#include "CtensorView.hpp"

#include "CtensorB.hpp"


namespace cnine{

    #ifdef _WITH_CUDA
    void Ctensor2view_accumulator_cu(const Ctensor2_view& r, const Ctensor2_view& x, const Rmask1& mask, const cudaStream_t& stream);
    #endif 


class Aggregator{
public:


  Aggregator(const Ctensor2_view& r, const Ctensor2_view& x, const Rmask1& mask){
    if(r.dev==0){
      assert(x.dev==0);
      for(auto it: mask.lists){
	auto t=r.slice0(it.first);
	auto& lst=it.second;
	for(int i=0; i<lst.size(); i++){
	  t.add(x.slice0(lst[i].first),lst[i].second);
	  //cout<<lst[i].first<<":"<<x.slice0(lst[i].first)<<endl;
	}
      }
    }
    if(r.dev==1){
#ifdef _WITH_CUDA
      cudaStream_t stream;
      CUDA_SAFE(cudaStreamCreate(&stream));
      Ctensor2view_accumulator_cu(r,x,mask,stream);
      CUDA_SAFE(cudaStreamSynchronize(stream));
      CUDA_SAFE(cudaStreamDestroy(stream));
#else
      CNINE_NOCUDA_ERROR;
#endif
    }  
  }


  Aggregator(const Ctensor3_view& r, const Ctensor3_view& x, const Rmask1& mask){
    Aggregator(r.fuse12(),x.fuse12(),mask);
  }


  Aggregator(const Ctensor4_view& r, const Ctensor4_view& x, const Rmask1& mask){
    Aggregator(r.fuse23(),x.fuse23(),mask);
  }

  Aggregator(const CtensorView& r, const CtensorView& x, const Rmask1& mask){
    Aggregator(r.fuse0X(),x.fuse0X(),mask);
  }
  Aggregator(const CtensorD_view& r, const CtensorD_view& x, const Rmask1& mask){
    Aggregator(r.fuse0X(),x.fuse0X(),mask);
  }

  
  /*
  Aggregator(CtensorB& r, const CtensorB& x, const Rmask1& mask){
    int k=r.dims.size();
    assert(x.dims.size()==k);
    if(k==2) Aggregator(r.view2(),x.view2(),mask);
    if(k==3) Aggregator(r.view3(),x.view3(),mask);
    if(k==4) Aggregator(r.view4(),x.view4(),mask);
  }
  */
  

};

}

#endif
