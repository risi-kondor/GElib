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


#ifndef _CellwiseUnaryCmap
#define _CellwiseUnaryCmap

#include "Cmaps2.hpp"


namespace cnine{

  class CellwiseUnaryCmap{
  public:
    
    int I;

    template<typename OP, typename ARR>
    CellwiseUnaryCmap(const OP& op, ARR& r, const ARR& x, const int add_flag=0){
      I=r.aasize;
      assert(x.aasize==I);
      if(r.dev==0){
	for(int i=0; i<I; i++){
	  decltype(x.get_cell(0)) t=r.cell(i);
	  if(add_flag==0) op.apply(t,x.cell(i));
	  else op.add(t,x.cell(i));
	}
      }
      if(r.dev==1){
	op.apply(*this,r,x,add_flag);
      }
    }

#ifdef _WITH_CUDA

    dim3 blockdims() const{
      return dim3(I);
    }

    __device__ thrust::tuple<int,int> operator()(const int i, const int j) const{
      return thrust::make_tuple(i,i);
    }

#endif 

  };


  // ---- Templates ------------------------------------------------------------------------------------------


  template<typename OP, typename ARR>
  ARR cellwise(const ARR& x){
    ARR r(x,x.adims,fill::raw);
    CellwiseUnaryCmap(OP(),r,x);
    return r;
  }

  template<typename OP, typename ARR>
  void add_cellwise(ARR& r, const ARR& x){
    CellwiseUnaryCmap(OP(),r,x,1);
  }

}


#endif 
