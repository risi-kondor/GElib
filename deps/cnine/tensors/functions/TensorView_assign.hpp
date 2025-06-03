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


#ifndef _cnine_TensorView_assign
#define _cnine_TensorView_assign

//#ifdef _WITH_CUDA
//#include <cuda.h>
//#include <cuda_runtime.h>
//#endif 


namespace cnine{

  template<typename TYPE> class TensorView;

  //template<> class TensorView<int>;

#ifdef _WITH_CUDA
  template<typename TYPE>
  void TensorView_assign_cu(const TensorView<TYPE>& r, const TensorView<TYPE>& x, const cudaStream_t& stream);
#endif 


  template<typename TYPE>
  void TensorView_assign(const TensorView<TYPE>& r, const TensorView<TYPE>& x){
    CNINE_ASSRT(r.get_dims()==x.get_dims());

    if(r.asize()==0) return; 
    if(r.is_contiguous() && r.get_strides()==x.get_strides()){
      TensorView_assign_copy(r,x);
      return;
    }

    auto [rp,xp]=r.co_scrunch(x);

    if(rp.get_dev()==0 && xp.get_dev()==0){
      TensorView_assign_loops(rp,xp);
      return;
    }

    if constexpr(!(std::is_same<TYPE,int>::value || 
	std::is_same<TYPE,float>::value || std::is_same<TYPE,double>::value)){
      CNINE_UNIMPL();
    }else{
      if(rp.get_dev()==1){
	if(xp.get_dev()==0) CUDA_STREAM(TensorView_assign_cu(rp,TensorView<TYPE>(xp,rp.get_dev()),stream));
	if(xp.get_dev()==1) CUDA_STREAM(TensorView_assign_cu(rp,xp,stream));
	return;
      }

      // TODO; refine this a lot!
      if(rp.get_dev()==0){
	if(rp.is_contiguous()){
	  TensorView<TYPE> z(MemArr<TYPE>(rp.get_strides().memsize(rp.get_dims()),xp.get_dev()),rp.get_dims(),rp.get_strides());
	  CUDA_STREAM(TensorView_assign_cu(z,xp,stream));
	  TensorView_assign(rp,z);
	}else{
	  TensorView<TYPE> z(xp,0);
	  TensorView_assign(rp,z);
	}
      }
    }
  }


  template<typename TYPE>
  void TensorView_assign_copy(const TensorView<TYPE>& r, const TensorView<TYPE>& x){
    if(r.get_dev()==0){
      if(x.get_dev()==0) std::copy(x.mem(),x.mem()+x.memsize(),r.mem());
      if(x.get_dev()==1) CUDA_SAFE(cudaMemcpy(r.mem(),x.mem(),x.memsize()*sizeof(TYPE),cudaMemcpyDeviceToHost));
    }
    if(r.get_dev()==1){
      if(x.get_dev()==0) CUDA_SAFE(cudaMemcpy(r.mem(),x.mem(),x.memsize()*sizeof(TYPE),cudaMemcpyHostToDevice));
      if(x.get_dev()==1) CUDA_SAFE(cudaMemcpy(r.mem(),x.mem(),x.memsize()*sizeof(TYPE),cudaMemcpyDeviceToDevice));  
    }      
  }


  template<typename TYPE>
  void TensorView_assign_loops(const TensorView<TYPE>& r, const TensorView<TYPE>& x){
    int D=r.ndims();
    TYPE* rarr=r.get_arr();
    TYPE* xarr=x.get_arr();
    bool copylast= r.get_strides().last()==1 && x.get_strides().last()==1 && r.get_dims().last()>16;

    if(D==1){
      int n0=r.dim(0);
      int rs0=r.get_strides()(0);
      int xs0=x.get_strides()(0);
      for(int i0=0; i0<n0; i0++)
	rarr[i0*rs0]=xarr[i0*xs0];
    }

    if(D==2){
      int n0=r.dim(0);
      int rs0=r.get_strides()(0);
      int xs0=x.get_strides()(0);
      int n1=r.dim(1);
      int rs1=r.get_strides()(1);
      int xs1=x.get_strides()(1);

      if(copylast){
	for(int i0=0; i0<n0; i0++)
	  std::copy(xarr+i0*xs0,xarr+i0*xs0+n1*xs0,rarr+i0*rs0);
      }else{
	for(int i0=0; i0<n0; i0++)
	  for(int i1=0; i1<n1; i1++)
	    rarr[i0*rs0+i1*rs1]=xarr[i0*xs0+i1*xs1];
      }
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

      if(copylast){
	for(int i0=0; i0<n0; i0++)
	  for(int i1=0; i1<n1; i1++)
	    std::copy(xarr+i0*xs0+i1*xs1,xarr+i0*xs0+i1*xs1+n2,rarr+i0*rs0+i1*rs1);
      }else{
	for(int i0=0; i0<n0; i0++)
	  for(int i1=0; i1<n1; i1++)
	    for(int i2=0; i2<n2; i2++)
	    rarr[i0*rs0+i1*rs1+i2*rs2]=xarr[i0*xs0+i1*xs1+i2*xs2];
      }
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

      if(copylast){
	for(int i0=0; i0<n0; i0++)
	  for(int i1=0; i1<n1; i1++)
	    for(int i2=0; i2<n2; i2++)
	      std::copy(xarr+i0*xs0+i1*xs1+i2*xs2,xarr+i0*xs0+i1*xs1+i2*xs2+n3,rarr+i0*rs0+i1*rs1+i2*rs2);
      }else{
	for(int i0=0; i0<n0; i0++)
	  for(int i1=0; i1<n1; i1++)
	    for(int i2=0; i2<n2; i2++)
	      for(int i3=0; i3<n3; i3++)
		rarr[i0*rs0+i1*rs1+i2*rs2+i3*rs3]=xarr[i0*xs0+i1*xs1+i2*xs2+i3*xs3];
      }
    }

    if(D>4){
      r.for_each([&](const Gindex& ix, TYPE& v) {v=x(ix);});
    }

  }

}

#endif 
