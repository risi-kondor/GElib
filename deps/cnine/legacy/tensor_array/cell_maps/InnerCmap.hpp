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


#ifndef _InnerCmap
#define _InnerCmap

#include "Cmaps2.hpp"


namespace cnine{

  class InnerCmap: public Masked2_cmap{
  public:

    int I;

    template<typename OP, typename ARR>
    InnerCmap(const OP& op, ARR& r, const ARR& x, const ARR& y, const int add_flag=0){
      I=x.get_aasize();
      assert(y.get_aasize()==I);
      if(r.dev==0){
	decltype(r.get_cell(0)) t=r.cell(0);
       	for(int i=0; i<I; i++)
	  op.apply(t,x.cell(i),y.cell(i),add_flag);
      }
      if(r.dev==1){
	// op.accumulate(*this,r,x,y,add_flag);
      }
    }

    /*
    dim3 blockdims() const{
      return dim3(I);
    }

    __device__ thrust::tuple<int,int,int> operator()(const int i, const int j, const int k) const{
      return thrust::make_tuple(0,k,k);
    }
    */

#ifdef _WITH_CUDA

    dim3 blockdims() const{
      return dim3(1);
    }

    __device__ int n_accum(const int b) const{
      return I;
    }

    __device__ int target(const int b) const{
      return 0;
    }

    __device__ int lst_ptr(const int b) const{
      return 0;
    }

    __device__ thrust::tuple<int,int> source(const int lst, const int b, const int i) const{
      return thrust::make_tuple(i,i);
    }

#endif 

  };


  // ---- Templates ------------------------------------------------------------------------------------------


  template<typename OP, typename ARR>
  ARR inner(const ARR& x, const ARR& y){
    ARR r=ARR::zeros_like(x,dims(1));
    InnerCmap(OP(),r,x,y);
    return r;
  }

  template<typename OP, typename ARR>
  void add_inner(ARR& r, const ARR& x, const ARR& y){
    InnerCmap(OP(),r,x,y,1);
  }

}

#endif 



