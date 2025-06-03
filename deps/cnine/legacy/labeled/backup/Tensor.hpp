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


#ifndef _CnineTensor
#define _CnineTensor

#include "Cnine_base.hpp"
#include "CnineObject.hpp"
#include "Gdims.hpp"
#include "Gstrides.hpp"
#include "Gtensor.hpp"

//#include "Rtensor2_view.hpp"
//#include "Rtensor3_view.hpp"
//#include "Rtensor4_view.hpp"
//#include "Rtensor5_view.hpp"
//#include "Rtensor6_view.hpp"
//#include "Rtensor7_view.hpp"
//#include "RtensorView.hpp"
#include "TensorView.hpp"

#ifdef _WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif 

#ifdef _WITH_CUBLAS
#include <cublas_v2.h>
extern cublasHandle_t cnine_cublas;
#endif 


namespace cnine{

  template<typename TYPE>
  class Tensor{
  public:

    Gdims dims;
    Gstrides strides;
    int k;
    int memsize=0;
    int dev=0;

    bool is_view=false;
    bool regular=true;

    TYPE* arr=nullptr;

  public:

    //RtensorA(){}

    ~Tensor(){
      deallocate();
    }

    string classname() const{
      return "Tensor";
    }

    string describe() const{
      return "Tensor"+dims.str();
    }

    //Tensor():
    //Tensor(Gdims({0})){}

  private: 

    void deallocate(){
     if(is_view) return;
      if(dev==0 && arr) {delete[] arr;}
      if(dev==1 && arr) {CUDA_SAFE(cudaFree(arrg));}
    }


  public: // ---- Constructors ------------------------------------------------------------------------------


    Tensor(const Gdims& _dims, const int _dev=0): 
      dims(_dims), 
      strides(_dims), 
      k(_dims.size()), 
      memsize(_dims.asize()),
      dev(_dev){
      allocate();
    }

    Tensor(const Gdims& _dims, const fill_noalloc& dummy, const int _dev=0): 
      dims(_dims), 
      strides(_dims), 
      k(_dims.size()), 
      dev(_dev){}


    Tensor(const Gdims& _dims, const Gstrides& _strides, const int _memsize, const bool _regular, const int _dev=0):
      dims(_dims), 
      strides(_strides), 
      k(_dims.size()), 
      memsize(_memsize), 
      dev(_dev), 
      regular(_regular){
      allocate();
    }

    Tensor(const fill_noalloc& dummy, const Gdims& _dims, const Gstrides& _strides, const int _memsize, const bool _regular, const int _dev=0):
      dims(_dims), 
      strides(_strides), 
      k(_dims.size()), 
      memsize(_memsize), 
      dev(_dev), 
      regular(_regular){
    }


  private:

    void allocate(){
      CPUCODE(arr=new TYPE[memsize];);
      GPUCODE(CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(TYPE))););
    }


  public: // ---- Filled constructors -----------------------------------------------------------------------


  public: // ---- Copying -----------------------------------------------------------------------------------


    Tensor(const Tensor<TYPE>& x): 
      Tensor(x.dims,x.strides,x.memsize,x.regular,x.dev){
      CNINE_COPY_WARNING();
      copy_as_blob_from(x);
    }
        
    Tensor(const Tensor<TYPE>& x, const nowarn_flag& dummy): 
      Tensor(x.dims,x.strides,x.memsize,x.regular,x.dev){
      copy_as_blob_from(x);
    }
        
    Tensor(const Tensor<TYPE>& x, const int _dev): 
      Tensor(x.dims,x.strides,x.memsize,x.regular,_dev){
      copy_as_blob_from(x);
    }
        
    Tensor(const Tensor<TYPE>& x, const view_flag& dummy){
      k=x.k; 
      dims=x.dims; 
      strides=x.strides; 
      dev=x.dev; 
      memsize=x.memsize; 
      arr=x.arr;
      is_view=true;
    }
        
    Tensor(Tensor&& x): 
      Tensor(x.fill_noalloc(),x.dims,x.strides,x.memsize,x.regular,x.dev){
      CNINE_MOVE_WARNING();
      arr=x.arr; x.arr=nullptr; 
      is_view=x.is_view;
    }

    Tensor& operator=(const Tensor& x){
      CNINE_ASSIGN_WARNING();

      if(dims==x.dims && regular && x.regular){
	copy_as_blob_from(x);
	return *this;
      }
      
      deallocate();
      k=x.k; 
      dims=x.dims; 
      strides=x.strides; 
      memsize=x.memsize; 
      regular=x.regular;
      is_view=false;
      dev=x.dev;
      allocate();
      copy_as_blob_from(x);
      return *this;
    }


    Tensor& operator=(Tensor&& x){
      CNINE_MOVEASSIGN_WARNING();
      k=x.k; dims=x.dims; strides=x.strides; dev=x.dev; 
      memsize=x.memsize; 
      deallocate();
      is_view=x.is_view;
      arr=x.arr; x.arr=nullptr; 
      return *this;
    }


  private:

    void copy_as_blob_from(const Tensor<TYPE>& x){
      PTENS_ASSRT(memsize==x.memsize);
      if(dev==0){
	if(x.dev==0) std::copy(x.arr,x.arr+memsize,arr);
	if(x.dev==1) CUDA_SAFE(cudaMemcpy(arr,x.arr,memsize*sizeof(TYPE),cudaMemcpyDeviceToHost)); 
      }
      if(dev==1){
	if(x.dev==0) CUDA_SAFE(cudaMemcpy(arr,x.arr,memsize*sizeof(TYPE),cudaMemcpyHostToDevice));
	if(x.dev==1) CUDA_SAFE(cudaMemcpy(arr,x.arr,memsize*sizeof(TYPE),cudaMemcpyDeviceToDevice));  
      }
    }


  public: // ---- Transport -----------------------------------------------------------------------------------


    /*
    Tensor& move_to(const int _dev){
      CNINE_DEVICE_VALID(_dev);

      if(_dev==0){
	if(dev==0) return *this;
 	delete[] arr;
	arr=new float[memsize];
	CUDA_SAFE(cudaMemcpy(arr,arrg,asize*sizeof(float),cudaMemcpyDeviceToHost));  
	CUDA_SAFE(cudaFree(arrg));
	const_cast<Tensor<TYPE>*>(this)->arrg=nullptr;
	dev=0;
	return *this;
      }

      if(_dev>0){
	if(dev==_dev) return *this;
	if(arrg) CUDA_SAFE(cudaFree(arrg));
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
	CUDA_SAFE(cudaMemcpy(arrg,arr,asize*sizeof(float),cudaMemcpyHostToDevice));  
	delete[] arr;
	const_cast<Tensor<TYPE>*>(this)->arr=nullptr;
	dev=_dev;
	return *this;
      }
      
      return *this;
    }
    */

    /*
    RtensorA& move_to(const device& _dev){
      return move_to_device(_dev.id());
    }
    
    RtensorA to(const device& _dev) const{
      return RtensorA(*this,_dev.id());
    }

    RtensorA to_device(const int _dev) const{
      return RtensorA(*this,_dev);
    }
    */


  };


}

#endif
