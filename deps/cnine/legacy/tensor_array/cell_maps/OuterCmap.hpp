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


#ifndef _OuterCmap
#define _OuterCmap

#include "Cmaps2.hpp"


namespace cnine{

  class OuterCmap: public Direct_cmap{ // public Cmap_base, 
  public:

    int I,J;
    int n;

    template<typename OP, typename ARR>
    OuterCmap(const OP& op, ARR& r, const ARR& x, const ARR& y, const int add_flag=0){
      I=x.get_aasize();
      J=y.get_aasize();
      n=y.get_aasize();
      assert(r.get_aasize()==I*J);

      if(r.dev==0){
       	for(int i=0; i<I; i++)
	  for(int j=0; j<J; j++){
	    decltype(x.get_cell(0)) t=r.cell(i*J+j);
	    op.apply(t,x.cell(i),y.cell(j),add_flag);
	  }
      }
      if(r.dev==1){
	op.apply(*this,r,x,y,add_flag);
      }
    }

#ifdef _WITH_CUDA

    dim3 blockdims() const{
      return dim3(I,J);
    }

    __device__ thrust::tuple<int,int,int> operator()(const int i, const int j, const int k) const{
      //printf("%d\n",n);
      return thrust::make_tuple(i*n+j,i,j);
    }

#endif 

  };



  // ---- Templates ------------------------------------------------------------------------------------------


  template<typename OP, typename ARR>
  ARR outer(const ARR& x, const ARR& y){
    ARR r=ARR::raw_like(x,dims(x.get_adim(0),y.get_adim(0)));
    OuterCmap(OP(),r,x,y);
    return r;
  }

  template<typename OP, typename ARR, typename ARG0>
  ARR outer(const ARR& x, const ARR& y, const ARG0& arg0){
    ARR r(x,dims(x.get_adim(0),y.get_adim(0)),fill::raw);
    OuterCmap(OP(arg0),r,x,y);
    return r;
  }


  template<typename OP, typename ARR>
  void add_outer(ARR& r, const ARR& x, const ARR& y){
    OuterCmap(OP(),r,x,y,1);
  }

  template<typename OP, typename ARR, typename ARG0>
  void add_outer(ARR& r, const ARR& x, const ARR& y, const ARG0& arg0){
    OuterCmap(OP(arg0),r,x,y,1);
  }


}

#endif 

