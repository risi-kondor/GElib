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


#ifndef _Convolve2Cmap
#define _Convolve2Cmap

#include "Cmaps2.hpp"


namespace cnine{

  class Convolve2Cmap: public Masked2_cmap{
  public:

    int I0,I1,J0,J1;
    int rs,xs,ys;

    template<typename OP, typename ARR>
    Convolve2Cmap(const OP& op, ARR& r, const ARR& x, const ARR& y, const int add_flag=0){
      assert(r.get_nadims()==2);
      assert(x.get_nadims()==2);
      assert(y.get_nadims()==2);
      
      I0=r.get_adim(0);
      I1=r.get_adim(1);
      J0=y.get_adim(0);
      J1=y.get_adim(1);
      assert(x.get_adim(0)==I0+J0-1);
      assert(x.get_adim(1)==I1+J1-1);
      rs=I1;
      xs=I1+J1-1;
      ys=J1;

      if(r.dev==0){
       	for(int i0=0; i0<I0; i0++){
	  for(int i1=0; i1<I1; i1++){
	    decltype(r.get_cell(0)) t=r.cell({i0,i1});
	    for(int j0=0; j0<J0; j0++)
	      for(int j1=0; j1<J1; j1++){
		op.apply(t,x.cell({i0+j0,i1+j1}),y.cell({j0,j1}),true);
	      }
	  }
	}
      }
      if(r.dev==1){
	//op.accumulate(*this,r,x,y);
      }
    }

#ifdef _WITH_CUDA

    dim3 blockdims() const{
      return dim3(I0*I1);
    }

    __device__ int n_accum(const int b) const{
      return J0*J1;
    }

    __device__ int target(const int b) const{
      return b;
    }

    __device__ int lst_ptr(const int b) const{
      return 0;
    }

    __device__ thrust::tuple<int,int> source(const int lst, const int b, const int j) const{
      const int i0=b/rs;
      const int i1=b%rs;
      const int j0=j/ys;
      const int j1=j%ys;
      return thrust::make_tuple((i0+j0)*xs+(i1+j1),j);
    }

#endif 

  };

  
  // ---- Templates ------------------------------------------------------------------------------------------

  
  template<typename OP, typename ARR>
  ARR convolve2(const ARR& x, const ARR& y){
    ARR r=ARR::zeros_like(x,dims(x.get_adim(0)-y.get_adim(0)+1,x.get_adim(1)-y.get_adim(1)+1));
    Convolve2Cmap(OP(),r,x,y);
    return r;
  }

  template<typename OP, typename ARR>
  void add_convolve2(ARR& r, const ARR& x, const ARR& y){
    Convolve2Cmap(OP(),r,x,y);
  }

}


#endif 


