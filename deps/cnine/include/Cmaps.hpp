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


#ifndef _Cnine_Cmaps
#define _Cnine_Cmaps

#include "Cnine_base.hpp"

#ifdef _WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/tuple.h>
#endif 


namespace cnine{


  class CellwiseICmap{
  public:
    
    int I;

    template<typename OP, typename ARR>
    CellwiseICmap(const OP& op, ARR& r){
      I=r.aasize;
      if(r.dev==0){
	for(int i=0; i<I; i++){
	  decltype(r.get_cell(0)) t=r.cell(i);
	  op(t);
	}
      }
      if(r.dev==1){
	op(*this,r);
      }
    }

    #ifdef _WITH_CUDA

    dim3 blockdims() const{
      return dim3(I);
    }

    __device__ int  operator()(const int i, const int j, const int k) const{
      return i;
    }
    #endif 

  };


  class ScatterICmap{
  public:
    
    int I;

    template<typename OP, typename ARR, typename OBJ>
    ScatterICmap(const OP& op, ARR& r, const OBJ& x){
      I=r.aasize;
      assert(x.aasize==I);
      if(r.dev==0){
	assert(x.dev==1);
	for(int i=0; i<I; i++){
	  decltype(r.get_cell(0)) t=r.cell(i);
	  op(t,x.get_value_at(x));
	}
      }
      if(r.dev==1){
	op(*this,r);
      }
    }

    #ifdef _WITH_CUDA

    dim3 blockdims() const{
      return dim3(I);
    }

    __device__ int  operator()(const int i, const int j, const int k) const{
      return i;
    }
    #endif 

  };


  // ---- Unary Cellmaps ------------------------------------------------------------------------------------


  class CellwiseUCmap{
  public:
    
    int I;

    template<typename OP, typename ARR>
    CellwiseUCmap(const OP& op, ARR& r, const ARR& x){
      I=r.aasize;
      assert(x.aasize==I);
      if(r.dev==0){
	assert(x.dev==1);
	for(int i=0; i<I; i++){
	  decltype(x.get_cell(0)) t=r.cell(i);
	  op(t,x.cell(i));
	}
      }
      if(r.dev==1){
	op(*this,r,x);
      }
    }

    #ifdef _WITH_CUDA

    dim3 blockdims() const{
      return dim3(I);
    }

    __device__ thrust::tuple<int,int> operator()(const int i, const int j, const int k) const{
      return thrust::make_tuple(i,i);
    }
    #endif 

  };


  class BroadcastUCmap{
  public:

    int I;
    
    template<typename OP, typename ARR>
    BroadcastUCmap(const OP& op, ARR& r, const ARR& x){
      I=r.aasize;
      assert(x.aasize==1);
      if(r.dev==0){
	assert(x.dev==0);
	decltype(x.get_cell(0)) _x=x.cell(0);
	for(int i=0; i<I; i++){
	  decltype(x.get_cell(0)) t=r.cell(i);
	  op(t,_x);
	}
      }
      if(r.dev==1){
	op(*this,r,x);
      }
    }

    #ifdef _WITH_CUDA

    dim3 blockdims() const{
      return dim3(I);
    }

    __device__ thrust::tuple<int,int> operator()(const int i, const int j, const int k) const{
      return thrust::make_tuple(i,0);
    }
    #endif 

  };


  class ScatterUCmap{
  public:
    
    int I;

    template<typename OP, typename ARR, typename OBJ>
    ScatterUCmap(const OP& op, ARR& r, const ARR& x, const OBJ& C){
      I=r.aasize;
      assert(x.aasize==I);
      if(r.dev==0){
	assert(x.dev==1);
	for(int i=0; i<I; i++){
	  decltype(x.get_cell(0)) t=r.cell(i);
	  op(t,x.cell(i),C.get_value_at(i));
	}
      }
      if(r.dev==1){
	op(*this,r,x);
      }
    }

    #ifdef _WITH_CUDA

    dim3 blockdims() const{
      return dim3(I);
    }

    __device__ thrust::tuple<int,int> operator()(const int i, const int j, const int k) const{
      return thrust::make_tuple(i,i);
    }
    #endif 

  };


  // ---- Binary Cellmaps ------------------------------------------------------------------------------------


  class CellwiseBiCmap{
  public:
    
    int I;

    template<typename OP, typename ARR>
    CellwiseBiCmap(const OP& op, ARR& r, const ARR& x, const ARR& y){
      I=r.aasize;
      assert(x.aasize==I);
      assert(y.aasize==I);
      if(r.dev==0){
	for(int i=0; i<I; i++){
	  decltype(r.cell_view(0)) t=r.cell_view(i);
	  op(t,x.cell(i),y.cell(i));
	}
      }
      if(r.dev==1){
	op(*this,r,x,y);
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


  class BroadcastLeftBiCmap{
  public:

    int I;
    
    template<typename OP, typename ARR>
    BroadcastLeftBiCmap(const OP& op, ARR& r, const ARR& x, const ARR& y){
      I=r.aasize;
      assert(y.aasize==I);
      if(r.dev==0){
	for(int i=0; i<r.aasize; i++){
	  decltype(r.cell_view(0)) t=r.cell_view(i);
	  op(t,x.cell(0),y.cell(i));
	}
      }
      if(r.dev==1){
	op(*this,r,ARR(x),y);
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


  class BroadcastRightBiCmap{
  public:

    int I;

    template<typename OP, typename ARR>
    BroadcastRightBiCmap(const OP& op, ARR& r, const ARR& x, const ARR& y){
      I=r.aasize;
      assert(x.aasize==I);
      if(r.dev==0){
	for(int i=0; i<r.aasize; i++){
	  decltype(x.cell_view(0)) t=r.cell(i);
	  op(t,x.cell(i),y.cell(0));
	}
      }
      if(r.dev==1){
	op(*this,r,x,ARR(y));
      }
    }

    #ifdef _WITH_CUDA

    dim3 blockdims() const{
      return dim3(I);
    }

    __device__ thrust::tuple<int,int,int> operator()(const int i, const int j, const int k) const{
      return thrust::make_tuple(i,i,0);
    }
    #endif 

  };


  class InnerBiCmap{
  public:

    int I;

    template<typename OP, typename ARR>
    InnerBiCmap(const OP& op, ARR& r, const ARR& x, const ARR& y){
      int I=x.aasize;
      assert(y.aasize==I);
      if(r.dev==0){
	decltype(r.cell_view(0)) t=r.cell_view(0);
       	for(int i=0; i<I; i++)
	  op(t,x.cell(i),y.cell(i));
      }
      if(r.dev==1){
	op(*this,r,x,y);
      }
    }

#ifdef _WITH_CUDA

    dim3 blockdims() const{
      return dim3(I);
    }

    __device__ thrust::tuple<int,int,int> operator()(const int i, const int j, const int k) const{
      return thrust::make_tuple(0,k,k);
    }

#endif 

  };


  class OuterBiCmap{
  public:

    int I,J;
    int n;

    template<typename OP, typename ARR>
    OuterBiCmap(const OP& op, ARR& r, const ARR& x, const ARR& y){
      I=x.aasize;
      J=y.aasize;
      n=y.aasize;
      if(r.dev==0){
	assert(r.aasize==I*J);
       	for(int i=0; i<I; i++)
	  for(int j=0; j<J; j++){
	    decltype(r.cell_view(0)) t=r.cell_view(i*J+j);
	    op(t,x.cell(i),y.cell(j));
	  }
      }
      if(r.dev==1){
	op(*this,r,x,y);
      }
    }

#ifdef _WITH_CUDA

    dim3 blockdims() const{
      return dim3(I,J);
    }

    __device__ thrust::tuple<int,int,int> operator()(const int i, const int j, const int k) const{
      return thrust::make_tuple(i*n+j,i,j);
    }
#endif 

  };


  class MprodBiCmap{
  public:

    int I,J;

    template<typename OP, typename ARR>
    MprodBiCmap(const OP& op, ARR& r, const ARR& x, const ARR& y){
      I=r.aasize;
      J=y.aasize;
      assert(x.aasize==I*J);
      if(r.dev==0){
       	for(int i=0; i<I; i++){
	  decltype(r.cell_view(0)) t=r.cell_view(i);
	  for(int j=0; j<J; j++){
	    op(t,x.cell(i*J+j),y.cell(j));
	  }
	}
      }
      if(r.dev==1){
	op(*this,r,x,y);
      }
    }

#ifdef _WITH_CUDA

    dim3 blockdims() const{
      return dim3(I,1,J);
    }

    __device__ thrust::tuple<int,int,int> operator()(const int i, const int j, const int k) const{
      return thrust::make_tuple(i,i*J+k,k);
    }

#endif 

  };


  class Convolve1BiCmap{
  public:

    int I,J;

    template<typename OP, typename ARR>
    Convolve1BiCmap(const OP& op, ARR& r, const ARR& x, const ARR& y){
      I=r.aasize;
      J=y.aasize;
      assert(x.aasize==I+J);
      if(r.dev==0){
       	for(int i=0; i<I; i++){
	  decltype(r.cell_view(0)) t=r.cell_view(i);
	  for(int j=0; j<J; j++){
	    op(t,x.cell(i+j),y.cell(j));
	  }
	}
      }
      if(r.dev==1){
	op(*this,r,x,y);
      }
    }

#ifdef _WITH_CUDA

    dim3 blockdims() const{
      return dim3(I,1,J);
    }

    __device__ thrust::tuple<int,int,int> operator()(const int i, const int j, const int k) const{
      return thrust::make_tuple(i,i+k,k);
    }

#endif 

  };


  class Convolve2BiCmap{
  public:

    int I,J;
    int rs,xs,ys;

    template<typename OP, typename ARR>
    Convolve2BiCmap(const OP& op, ARR& r, const ARR& x, const ARR& y){
      assert(r.adims.size()==2);
      assert(x.adims.size()==2);
      assert(y.adims.size()==2);
      
      const int I0=r.get_adim(0);
      const int I1=r.get_adim(1);
      const int J0=y.get_adim(0);
      const int J1=y.get_adim(1);
      assert(x.get_adim(0)==I0+J0-1);
      assert(x.get_adim(1)==I1+J1-1);
      rs=I1;
      xs=I1+J1-1;
      ys=J1;
      I=I0*I1;
      J=J0*J1;

      if(r.dev==0){
       	for(int i0=0; i0<I0; i0++){
	  for(int i1=0; i1<I1; i1++){
	    decltype(r.cell_view(0)) t=r.cell_view({i0,i1});
	    for(int j0=0; j0<J0; j0++)
	      for(int j1=0; j1<J1; j1++){
		op(t,x.cell({i0+j0,i1+j1}),y.cell({j0,j1}));
	      }
	  }
	}
      }
      if(r.dev==1){
	op(*this,r,x,y);
      }
    }

#ifdef _WITH_CUDA

    dim3 blockdims() const{
      return dim3(I,0,J);
    }

    __device__ thrust::tuple<int,int,int> operator()(const int i, const int j, const int k) const{
      return thrust::make_tuple(i,(i/rs+k/ys)*xs+i%rs+k%ys,k);
    }

#endif 

  };


}

#endif 
