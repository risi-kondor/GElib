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


#ifndef _accumulate_cmap
#define _accumulate_cmap

#include "Cmaps2.hpp"
#include "CellMask2r.hpp"


namespace cnine{

  class accumulate_cmap: public Masked2_cmap{
  public:
    
    int I;
    const CellMask2r& mask;
    int n=0;
    int* arrg=nullptr;

    template<typename OP, typename ARR>
    accumulate_cmap(const OP& op, const CellMask2r& _mask, ARR& r, const ARR& x, const ARR& y, const int add_flag=0):
      mask(_mask){
      assert(add_flag);
      if(r.dev==0){
	for(auto it: mask.lists){
	  decltype(x.get_cell(0)) t=r.cell(it.first);
	  const CellTlist2& lst=*it.second;
	  if(lst.size()>0) 
	    op.apply(t,x.cell(lst[0].first),y.cell(lst[0].second),add_flag);
	  for(int i=1; i<lst.size(); i++)
	    op.apply(t,x.cell(lst[i].first),y.cell(lst[i].second),true);
	  //for(auto p:lst){
	  //op.apply(t,x.cell(p.first),y.cell(p.second));
	  //}
	}
      }
      if(r.dev==1){
	mask.prepare(1);
	n=mask.n;
	arrg=mask.arrg;
	op.accumulate(*this,r,x,y,add_flag);
      }
    }

    template<typename OP, typename ARR>
    accumulate_cmap(const OP& op, const CellMask2r& mask, ARR& r, const ARR& x, const ARR& y, 
      const cmap_add& dummy):
      accumulate_cmap(op,mask,r,x,y,true){}


#ifdef _WITH_CUDA

    dim3 blockdims() const{
      return n;
    }

    __device__ int n_accum(const int b) const{
      return arrg[arrg[b]+1];
    }

    __device__ int target(const int b) const{
      return arrg[arrg[b]];
    }

    __device__ int lst_ptr(const int b) const{
      return arrg[b]+2;
    }

    __device__ thrust::tuple<int,int> source(const int lst, const int b, const int j) const{
      return thrust::make_tuple(arrg[lst+2*j],arrg[lst+2*j+1]);
    }

    // dummy function should be possible to eliminate
    //__device__ thrust::tuple<int,int,int> operator()(const int i, const int j, const int k) const{
    //return thrust::make_tuple(i,i,i);
    //}


#endif 

  };


  template<typename OP, typename ARR>
  void add_accumulate(const CellMask2r& mask, ARR& r, const ARR& x, const ARR& y){
    OP op;
    accumulate_cmap(op,mask,r,x,y,cmap_add());
  }

  template<typename OP, typename ARR>
  ARR accumulate(const CellMask2r& mask, const ARR& x, const ARR& y){
    OP op;
    ARR r(x,x.adims,fill::zero); // todo
    accumulate_cmap(op,mask,r,x,y);
    return r;
  }


}

#endif 
