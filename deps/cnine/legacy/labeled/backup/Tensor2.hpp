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
#include "Gdims.hpp"
#include "Gstrides.hpp"
#include "Gindex.hpp"
#include "MemArr.hpp"

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

    MemArr<TYPE> arr;
    Gdims dims;
    Gstrides strides;
    int dev;
    bool regular=true;


  public: // ---- Constructors ------------------------------------------------------------------------------


    Tensor(const Gdims& _dims, const int _dev=0): 
      arr(_dims.total(),_dev),
      dims(_dims), 
      strides(_dims), 
      dev(_dev){
    }

    Tensor(const Gdims& _dims, const fill_zero& dummy, const int _dev=0): 
      arr(_dims.total(),dummy,_dev),
      dims(_dims), 
      strides(_dims), 
      dev(_dev){
    }

    Tensor(const Gdims& _dims, const fill_sequential& dummy, const int _dev=0): 
      arr(_dims.total(),dummy,_dev),
      dims(_dims), 
      strides(_dims), 
      dev(_dev){
    }


  private:

    Tensor(MemArr<TYPE>&& _arr, const Gdims& _dims, const Gstrides& _strides, const int _regular, const int _dev=0):
      arr(std::move(_arr)),
      dims(_dims), 
      strides(_strides), 
      dev(_dev),
      regular(_regular){}


  public: // ---- Named constructors ------------------------------------------------------------------------


    static Tensor<TYPE> zero(const Gdims& _dims, const int _dev=0){
      return Tensor<TYPE>(_dims,fill_zero(),_dev);
    }

    static Tensor<TYPE> sequential(const Gdims& _dims, const int _dev=0){
      return Tensor<TYPE>(_dims,fill_sequential(),_dev);
    }



  public: // ---- Copying -----------------------------------------------------------------------------------


    Tensor(const Tensor<TYPE>& x):
      arr(x.arr),
      dims(x.dims),
      strides(x.strides),
      dev(x.dev),
      regular(x.regular){
      CNINE_COPY_WARNING();
    }
        
    Tensor(const Tensor<TYPE>& x, const nowarn_flag& dummy):
      arr(x.arr),
      dims(x.dims),
      strides(x.strides),
      dev(x.dev),
      regular(x.regular){
    }
        
    Tensor(Tensor<TYPE>&& x):
      arr(std::move(x.arr)),
      dims(x.dims),
      strides(x.strides),
      dev(x.dev),
      regular(x.regular){
      CNINE_MOVE_WARNING();
    }

    Tensor& operator=(const Tensor& x){
      CNINE_ASSRT(dims==x.dims);
      CNINE_ASSIGN_WARNING();
      if(regular&&x.regular){
	arr=x.arr;
      }else{
	cout<<"not regular"<<endl;
      }
      return *this;
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


  public: // ---- Access -------------------------------------------------------------------------------------


    int ndims() const{
      return dims.size();
    }

    bool is_regular() const{
      return regular;
    }

    int asize() const{
      return dims.asize();
    }

    //int memsize() const{
    //return strides.memsize(dims);
    //}


  public: // ---- Getters ------------------------------------------------------------------------------------


    TYPE operator()(const Gindex& ix) const{
      CNINE_CHECK_RANGE(dims.check_in_range(ix,string(__PRETTY_FUNCTION__)));
      return arr[ix(strides)];
    }

    TYPE operator()(const int i0) const{
      CNINE_CHECK_RANGE(dims.check_in_range(i0,string(__PRETTY_FUNCTION__)));
      return arr[strides.offs(i0)];
    }

    TYPE operator()(const int i0, const int i1) const{
      CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,string(__PRETTY_FUNCTION__)));
      return arr[strides.offs(i0,i1)];
    }

    TYPE operator()(const int i0, const int i1, const int i2) const{
      CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,i2,string(__PRETTY_FUNCTION__)));
      return arr[strides.offs(i0,i1,i2)];
    }

    TYPE operator()(const int i0, const int i1, const int i2, const int i3) const{
      CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,i2,i3,string(__PRETTY_FUNCTION__)));
      return arr[strides.offs(i0,i1,i2,i3)];
    }


  public: // ---- Setters ------------------------------------------------------------------------------------


    void set(const Gindex& ix, const TYPE x){
      CNINE_CHECK_RANGE(dims.check_in_range(ix,string(__PRETTY_FUNCTION__)));
      arr[ix(strides)]=x;
    }

    void set(const int i0, const TYPE x){
      CNINE_CHECK_RANGE(dims.check_in_range(i0,string(__PRETTY_FUNCTION__)));
      arr[strides.offs(i0)]=x;
    }

    void set(const int i0, const int i1,  const TYPE x){
      CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,string(__PRETTY_FUNCTION__)));
      arr[strides.offs(i0,i1)]=x;
    }

    void set(const int i0, const int i1, const int i2, const TYPE x){
      CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,i2,string(__PRETTY_FUNCTION__)));
      arr[strides.offs(i0,i1,i2)]=x;
    }

    void set(const int i0, const int i1, const int i2, const int i3, const TYPE x){
      CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,i2,i3,string(__PRETTY_FUNCTION__)));
      arr[strides.offs(i0,i1,i2,i3)]=x;
    }


  public: // ---- Slices ------------------------------------------------------------------------------------


    Tensor<TYPE> slice(const int d, const int i){
      CNINE_CHECK_DIM(d,i);
      return Tensor<TYPE>(arr.offset(strides[d]*i),dims.remove(d),strides.remove(d),false,dev);
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "Tensor";
    }

    string describe() const{
      ostringstream oss;
      oss<<"Tensor"<<dims<<" ["<<strides<<"]"<<endl;
      return oss.str();
    }

    string str(const string indent="") const{
      CNINE_CPUONLY();
      ostringstream oss;

      if(ndims()==1){
	oss<<"[ ";
	for(int i0=0; i0<dims[0]; i0++)
	  oss<<(*this)(i0)<<" ";
	oss<<"]"<<endl;
	return oss.str();
      }

      if(ndims()==2){
	for(int i0=0; i0<dims[0]; i0++){
	  oss<<"[ ";
	  for(int i1=0; i1<dims[1]; i1++)
	    oss<<(*this)(i0,i1)<<" ";
	  oss<<"]"<<endl;
	}
	return oss.str();
      }

      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Tensor<TYPE>& x){
      stream<<x.str(); return stream;
    }

  };


}

#endif
