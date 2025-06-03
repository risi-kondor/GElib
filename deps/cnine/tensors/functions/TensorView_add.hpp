/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2024, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#ifndef _cnine_TensorView_add
#define _cnine_TensorView_add

//#include "TensorView.hpp"


namespace cnine{

  template<typename TYPE> class TensorView;

#ifdef _WITH_CUDA
  // Forward declaration
  template<typename TYPE>
  void TensorView_add_cu(const TensorView<TYPE>& r, const TensorView<TYPE>& x, const cudaStream_t& stream);
#endif 


  template<typename TYPE>
  void TensorView_add(const TensorView<TYPE>& r, const TensorView<TYPE>& x){
    CNINE_ASSRT(r.get_dims()==x.get_dims());
    if(r.asize()==0) return; 

    //cout<<"----------------"<<endl;
    //cout<<r.dims<<" "<<x.dims<<endl;
    //cout<<r.strides<<" "<<x.strides<<endl;
    //cout<<"x="<<x<<endl;

    int dev=r.get_dev();
    CNINE_ASSRT(x.get_dev()==dev);

    if(r.is_contiguous() && r.get_strides()==x.get_strides()){
      if(dev==0){
	stdadd(x.get_arr(),x.get_arr()+x.memsize(),r.get_arr());
	return;
      }
      if(dev==1){
	if constexpr(std::is_same<TYPE,float>::value){
	  const float alpha=1.0; // todo, saxpy can work for non-contiguous too
	  CUBLAS_SAFE(cublasSaxpy(cnine_cublas, r.memsize(), &alpha, x.get_arr(), 1, r.get_arr(), 1));
	  return;
	}
	if constexpr(std::is_same<TYPE,double>::value){
	  const double alpha=1.0; // todo, daxpy can work for non-contiguous too
	  CUBLAS_SAFE(cublasDaxpy(cnine_cublas, r.memsize(), &alpha, x.get_arr(), 1, r.get_arr(), 1));
	  return;
	}
	if constexpr(std::is_same<TYPE,int>::value){
	  CUDA_STREAM(TensorView_add_cu(r,x,stream));
	  return;
	}
	CNINE_UNIMPL();
      }
      return;
    }

    auto [rp,xp]=r.co_scrunch(x);
    //cout<<rp.dims<<" "<<xp.dims<<endl;
    //cout<<rp.strides<<" "<<xp.strides<<endl;

    if(dev==0){
      TensorView_add_loops(rp,xp);
    }

    if(dev==1){// this is probably not going to work for non-contiguous tensors
      if constexpr(std::is_same<TYPE,int>::value || 
	std::is_same<TYPE,float>::value || 
	std::is_same<TYPE,double>::value){
	CUDA_STREAM(TensorView_add_cu(rp,xp,stream));
      }else{
	CNINE_UNIMPL();
      }
    }
  }


  template<typename TYPE>
  void TensorView_add_loops(const TensorView<TYPE>& r, const TensorView<TYPE>& x){
    int D=r.ndims();
    TYPE* rarr=r.get_arr();
    TYPE* xarr=x.get_arr();

    if(D==1){
      int n0=r.dim(0);
      int rs0=r.get_strides()(0);
      int xs0=x.get_strides()(0);
      for(int i0=0; i0<n0; i0++)
	rarr[i0*rs0]+=xarr[i0*xs0];
    }

    if(D==2){
      //cout<<r<<x<<endl;
      int n0=r.dim(0);
      int rs0=r.get_strides()(0);
      int xs0=x.get_strides()(0);
      int n1=r.dim(1);
      int rs1=r.get_strides()(1);
      int xs1=x.get_strides()(1);

      for(int i0=0; i0<n0; i0++)
	for(int i1=0; i1<n1; i1++)
	  rarr[i0*rs0+i1*rs1]+=xarr[i0*xs0+i1*xs1];
      //cout<<r<<endl;
    }

    if(D==3){
      int n0=r.dim(0);
      int rs0=r.get_strides()(0);
      int xs0=x.get_strides()(0);
      int n1=r.dim(1);
      int rs1=r.get_strides()(1);
      int xs1=x.get_strides()(1);
      int n2=r.dim(2);
      int rs2=r.get_strides()(2);
      int xs2=x.get_strides()(2);

      for(int i0=0; i0<n0; i0++)
	for(int i1=0; i1<n1; i1++)
	  for(int i2=0; i2<n2; i2++)
	    rarr[i0*rs0+i1*rs1+i2*rs2]+=xarr[i0*xs0+i1*xs1+i2*xs2];
    }

    if(D==4){
      int n0=r.dim(0);
      int rs0=r.get_strides()(0);
      int xs0=x.get_strides()(0);
      int n1=r.dim(1);
      int rs1=r.get_strides()(1);
      int xs1=x.get_strides()(1);
      int n2=r.dim(2);
      int rs2=r.get_strides()(2);
      int xs2=x.get_strides()(2);
      int n3=r.dim(3);
      int rs3=r.get_strides()(3);
      int xs3=x.get_strides()(3);

      for(int i0=0; i0<n0; i0++)
	for(int i1=0; i1<n1; i1++)
	  for(int i2=0; i2<n2; i2++)
	    for(int i3=0; i3<n3; i3++)
	      rarr[i0*rs0+i1*rs1+i2*rs2+i3*rs3]+=xarr[i0*xs0+i1*xs1+i2*xs2+i3*xs3];
    }

    if(D>4){
      r.for_each([&](const Gindex& ix, TYPE& v) {v+=x(ix);});
    }

  }

}

#endif 
