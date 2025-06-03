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


#ifndef _CellwiseBinaryCmap
#define _CellwiseBinaryCmap

#include "Cmaps2.hpp"


namespace cnine{

  class CellwiseBinaryCmap: public Direct_cmap{
  public:
    
    int I;

    CellwiseBinaryCmap(){
      I=1;
    }

    template<typename OP, typename ARR>
    CellwiseBinaryCmap(const OP& op, ARR& r, const ARR& x, const ARR& y, const int add_flag=0){
      I=r.get_aasize();
      assert(x.get_aasize()==I);
      assert(y.get_aasize()==I);
      if(r.dev==0){
	for(int i=0; i<I; i++){
	  decltype(x.get_cell(0)) t=r.cell(i);
	  op.apply(t,x.cell(i),y.cell(i),add_flag);
	}
      }
      if(r.dev==1){
	op.apply(*this,r,x,y,add_flag);
      }
    }
    
#ifdef _WITH_CUDA

    dim3 blockdims() const{
      return dim3(I);
    }

    __device__ thrust::tuple<int,int,int> operator()(const int i, const int j, const int k) const{
      return thrust::make_tuple(i,i,i);
    }

#endif 

  };


  // ---- Templates ------------------------------------------------------------------------------------------


  template<typename OP, typename ARR>
  ARR cellwise(const ARR& x, const ARR& y){
    ARR r=ARR::raw_like(x);
    CellwiseBinaryCmap(OP(),r,x,y);
    return r;
  }

  template<typename OP, typename ARR, typename ARG0>
  ARR cellwise(const ARR& x, const ARR& y, const ARG0& arg0){
    ARR r=ARR::raw_like(x);
    OP op(arg0);
    CellwiseBinaryCmap(op,r,x,y);
    return r;
  }

  template<typename OP, typename ARR, typename ARG0, typename ARG1>
  ARR cellwise(const ARR& x, const ARR& y, const ARG0& arg0, const ARG1& arg1){
    ARR r=ARR::raw_like(x);
    OP op(arg0,arg1);
    CellwiseBinaryCmap(op,r,x,y);
    return r;
  }



  template<typename OP, typename ARR>
  void add_cellwise(ARR& r, const ARR& x, const ARR& y){
    CellwiseBinaryCmap(OP(),r,x,y,1);
  }

  template<typename OP, typename ARR, typename ARG0>
  void add_cellwise(ARR& r, const ARR& x, const ARR& y, const ARG0& arg0){
    CellwiseBinaryCmap(OP(arg0),r,x,y,1);
  }


}

#endif 


