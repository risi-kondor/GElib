/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _cnineGatherSlices
#define _cnineGatherSlices

#include "Cnine_base.hpp"
#include "GatherMapB.hpp"
#include "GatherMapPack.hpp"
#include "WeightedGatherMapB.hpp"
#include "FixedkGatherMap.hpp"
#include "Ltensor.hpp"
#include "logged_timer.hpp"
#include "MultiLoop.hpp"


namespace cnine{

#ifdef _WITH_CUDA
  template<typename TYPE>
  void TensorView_gather_cu(TensorView<TYPE> r, TensorView<TYPE> x, const GatherMapB& gmap, const cudaStream_t& stream);
#endif 


  class GatherSlices{
  public:

    GatherSlices(){}

    template<typename TYPE>
    TensorView<TYPE> operator()(const TensorView<TYPE>& X, const GatherMapB& gmap, int in_dim, int out_dim=-1){
      if(out_dim==-1) out_dim=in_dim;
      CNINE_ASSRT(gmap.n_in==X.dim(in_dim));
      TensorView<TYPE> R(X.get_dims().remove(in_dim).insert(out_dim,gmap.n_out),0,X.get_dev());
      (*this)(R,X,gmap,in_dim,out_dim);
      return R;
    }

    template<typename TYPE>
    TensorView<TYPE> naive(const TensorView<TYPE>& X, const GatherMapB& gmap, int in_dim, int out_dim=-1){
      if(out_dim==-1) out_dim=in_dim;
      CNINE_ASSRT(gmap.n_in==X.dim(in_dim));
      TensorView<TYPE> R(X.get_dims().remove(in_dim).insert(out_dim,gmap.n_out),0,X.get_dev());
      naive(R,X,gmap,in_dim,out_dim);
      return R;
    }


  public: // ------------------------------------------------------------------------------------------------


    template<typename TYPE>
    void operator()(const TensorView<TYPE>& _R, const TensorView<TYPE>& _X, const GatherMapB& gmap, int in_dim, int out_dim=-1){
      if(out_dim==-1) out_dim=in_dim;
      auto p=_R.co_scrunch_except(_X,out_dim,in_dim);
      auto R(p.first);
      auto X(p.second);

      int k=R.ndims()-1;
      CNINE_ASSRT(R.dim(0)==gmap.n_out);
      CNINE_ASSRT(X.dim(0)==gmap.n_in);

      int dev=R.get_dev();
      CNINE_ASSRT(X.get_dev()==dev);
      
      if(dev==0){

	if(k==1){
	  int rs0=R.strides[0];
	  int rs1=R.strides[1];

	  int xs0=X.strides[0];
	  int xs1=X.strides[1];

	  int d1=X.dims[1];

	  if(rs1==1 & xs1==1){
	    gmap.for_each([&](const int i, const int j){
		stdadd(X.get_arr()+j*xs0,X.get_arr()+j*xs0+d1,R.get_arr()+i*rs0);});
	  }else{
	    gmap.for_each([&](const int i, const int j){
		for(int s=0; s<d1; s++)
		  R.arr[j*rs0+s*rs1]+=X.arr[i*xs0+j*rs1];
	      });
	  }
	  return;
	}

	if(k==2){
	  int rs0=R.strides[0];
	  int rs1=R.strides[1];
	  int rs2=R.strides[2];

	  int xs0=X.strides[0];
	  int xs1=X.strides[1];
	  int xs2=X.strides[2];

	  int d1=X.dims[1];
	  int d2=X.dims[2];

	  if(rs2==1 & xs2==1){
	    gmap.for_each([&](const int i, const int j){
		for(int i1=0; i1<d1; i1++)
		  stdadd(X.get_arr()+j*xs0+i1*xs1,X.get_arr()+j*xs0+i1*xs1+d2,R.get_arr()+i*rs0+i1*rs1);});
	  }else{
	    gmap.for_each([&](const int i, const int j){
		for(int i1=0; i1<d1; i1++)
		  for(int i2=0; i2<d2; i2++)
		    R.arr[j*rs0+i1*rs1+i2*rs2]+=X.arr[i*xs0+i1*xs1+i2*xs2];
	      });
	  }
	  return;
	}

	if(k==3){
	  int rs0=R.strides[0];
	  int rs1=R.strides[1];
	  int rs2=R.strides[2];
	  int rs3=R.strides[3];

	  int xs0=X.strides[0];
	  int xs1=X.strides[1];
	  int xs2=X.strides[2];
	  int xs3=X.strides[3];

	  int d1=X.dims[1];
	  int d2=X.dims[2];
	  int d3=X.dims[2];

	  if(rs3==1 && xs3==1){
	    gmap.for_each([&](const int i, const int j){
		for(int i1=0; i1<d1; i1++)
		  for(int i2=0; i2<d2; i2++)
		    stdadd(X.get_arr()+j*xs0+i1*xs1+i2*xs2,X.get_arr()+j*xs0+i1*xs1+i2*xs2+d3,
		      R.get_arr()+i*rs0+i1*rs1+i2*rs2);});
	  }else{
	    gmap.for_each([&](const int i, const int j){
		for(int i1=0; i1<d1; i1++)
		  for(int i2=0; i2<d2; i2++)
		    for(int i3=0; i3<d3; i3++)
		      R.arr[j*rs0+i1*rs1+i2*rs2+i3*rs3]+=X.arr[i*xs0+i1*xs1+i2*xs2+i3*xs3];
	      });
	  }
	  return;
	}

	CNINE_UNIMPL();

      }
      
      if(dev==1){
	if constexpr(std::is_same<TYPE,int>::value || 
	  std::is_same<TYPE,float>::value || 
	  std::is_same<TYPE,double>::value){
	  CUDA_STREAM(TensorView_gather_cu(R,X,gmap,stream));
	}else{
	  CNINE_UNIMPL();
	}
      }
    }

    template<typename TYPE>
    void naive(const TensorView<TYPE>& _R, const TensorView<TYPE>& _X, const GatherMapB& gmap, int in_dim, int out_dim=-1){
      if(out_dim==-1) out_dim=in_dim;

      int m=_R.ndims();
      CNINE_ASSRT(_X.ndims()==m);
      CNINE_ASSRT(m<5);

      Gdims xdims(m,fill_raw());
      GstridesB xstrides(m,fill_raw());
      xdims[0]=_X.dim(in_dim);
      xstrides[0]=_X.stride(in_dim);
      for(int i=0; i<in_dim; i++){
	xdims[i+1]=_X.dim(i);
	xstrides[i+1]=_X.stride(i);
      }
      for(int i=in_dim+1; i<m; i++){
	xdims[i]=_X.dim(i);
	xstrides[i]=_X.stride(i);
      }
      TensorView X(_X.arr,xdims,xstrides);

      Gdims rdims(m,fill_raw());
      GstridesB rstrides(m,fill_raw());
      rdims[0]=_R.dim(out_dim);
      rstrides[0]=_R.stride(out_dim);
      for(int i=0; i<out_dim; i++){
	rdims[i+1]=_R.dim(i);
	rstrides[i+1]=_R.stride(i);
      }
      for(int i=out_dim+1; i<m; i++){
	rdims[i]=_R.dim(i);
	rstrides[i]=_R.stride(i);
      }
      TensorView R(_R.arr,rdims,rstrides);

      if(m==1){
	gmap.for_each([&](const int i0, const int j0){
	    R.inc(i0,X(j0));});
      }

      if(m==2){
	gmap.for_each([&](const int i0, const int j0){
	    for(int i1=0; i1<X.dim(1); i1++)
	      R.inc(i0,i1,X(j0,i1));});
      }

      if(m==3){
	gmap.for_each([&](const int i0, const int j0){
	    for(int i1=0; i1<X.dim(1); i1++)
	      for(int i2=0; i2<X.dim(2); i2++)
		R.inc(i0,i1,i2,X(j0,i1,i2));});
      }

      if(m==4){
	gmap.for_each([&](const int i0, const int j0){
	    for(int i1=0; i1<X.dim(1); i1++)
	      for(int i2=0; i2<X.dim(2); i2++)
		for(int i3=0; i3<X.dim(3); i3++)
		  R.inc(i0,i1,i2,i3,X(j0,i1,i2,i3));});
      }

    }

  };



}

#endif 
