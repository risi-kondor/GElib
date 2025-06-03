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


#ifndef _AccumulateCmap
#define _AccumulateCmap

#include "Cmaps2.hpp"
#include "Rmask1.hpp"


namespace cnine{

  class AccumulateCmap{
  public:
    
    int I;
    const Rmask1& mask;
    //int n=0;
    //int* arrg=nullptr;

    template<typename OP>
    AccumulateCmap(const OP& op, const Ctensor2_view& r, const Ctensor2_view& x, const Rmask1& _mask, const bool add_flag=true):
      mask(_mask){
      assert(add_flag);

      if(r.dev==0){
	assert(x.dev==0);
	for(auto it: mask.lists){
	  auto t=r.slice0(it.first);
	  auto& lst=it.second;
	  if(lst.size()>0) 
	    op.apply(t,x.slice0(lst[0].first),lst[0].second,add_flag);
	  for(int i=1; i<lst.size(); i++)
	    op.apply(t,x.slice0(lst[i].first),lst[i].second);
	}
      }

      if(r.dev==1){
	mask.prepare(1);
	//n=mask.n;
	//arrg=mask.arrg;
	op.accumulate(*this,r,x,add_flag);
      }

    }

    /*
    template<typename OP, typename ARR>
    AccumulateCmap(const OP& op, const CellMask2r& mask, ARR& r, const ARR& x, const ARR& y, 
      const cmap_add& dummy):
      AccumulateCmap(op,mask,r,x,y,true){}
    */

#ifdef _WITH_CUDA

    dim3 blockdims() const{
      return 0; //TODO
    }

    __device__ int n_accum(const int b) const{
      return mask.arrg[mask.ptrg[b]+1];
    }

    __device__ int target(const int b) const{
      return mask.arrg[mask.ptrg[b]];
    }

    __device__ int lst_ptr(const int b) const{
      return mask.arrg[b]+2;
    }

    __device__ thrust::tuple<int,int> source(const int lst, const int b, const int j) const{
      return thrust::make_tuple(mask.arrg[lst+2*j],mask.arrg[lst+2*j+1]);
    }

    // dummy function should be possible to eliminate
    //__device__ thrust::tuple<int,int,int> operator()(const int i, const int j, const int k) const{
    //return thrust::make_tuple(i,i,i);
    //}


#endif 

  };


  /*
  template<typename OP, typename ARR>
  void add_accumulate(const CellMask2r& mask, ARR& r, const ARR& x, const ARR& y){
    OP op;
    AccumulateCmap(op,mask,r,x,y,cmap_add());
  }

  template<typename OP, typename ARR>
  ARR accumulate(const CellMask2r& mask, const ARR& x, const ARR& y){
    OP op;
    ARR r(x,x.adims,fill::zero); // todo
    AccumulateCmap(op,mask,r,x,y);
    return r;
  }
  */


}

#endif 
