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


#ifndef _BroadcastBinaryCmap
#define _BroadcastBinaryCmap

#include "Cmaps2.hpp"

namespace cnine{

  class BroadcastBinaryCmap: public Direct_cmap{
  public:

    int I;
    
    template<typename OP, typename ARR>
    BroadcastBinaryCmap(const OP& op, ARR& r, const decltype(r.get_cell(0))& x, const ARR& y, const int add_flag=0){
      r.get_aasize();
      assert(y.get_aasize()==I);
      if(r.dev==0){
	for(int i=0; i<I; i++){
	  decltype(r.get_cell(0)) t=r.cell(i);
	  op.apply(t,x,y.cell(i),add_flag);
	}
      }
      if(r.dev==1){
	//op.apply(*this,r,ARR(x),y,add_flag);
      }
    }

#ifdef _WITH_CUDA

    dim3 blockdims() const{
      return dim3(I);
    }

    __device__ thrust::tuple<int,int,int> operator()(const int i, const int j, const int k) const{
      return thrust::make_tuple(i,0,i);
    }

#endif 

  };


  // ---- Templates ------------------------------------------------------------------------------------------


  template<typename OP, typename OBJ, typename ARR>
  ARR broadcast(const OBJ& x, const ARR& y){
    ARR r=ARR::raw_like(x,y.get_adims());
    BroadcastBinaryCmap(OP(),r,x,y);
    return r;
  }

  template<typename OP, typename OBJ, typename ARR, typename ARG0>
  ARR broadcast(const OBJ& x, const ARR& y, const ARG0& arg0){
    ARR r=ARR::raw_like(x,y.get_adims());
    BroadcastBinaryCmap(OP(arg0),r,x,y);
    return r;
  }


  template<typename OP, typename OBJ, typename ARR>
  void add_broadcast(ARR& r, const OBJ& x, const ARR& y){
    BroadcastBinaryCmap(OP(),r,x,y,1);
  }

  template<typename OP, typename OBJ, typename ARR, typename ARG0>
  void add_broadcast(ARR& r, const OBJ& x, const ARR& y, const ARG0& arg0){
    BroadcastBinaryCmap(OP(arg0),r,x,y,1);
  }

}

#endif


