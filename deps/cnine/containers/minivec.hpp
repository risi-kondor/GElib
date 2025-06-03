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

#ifndef _minivec
#define _minivec

#include "Cnine_base.hpp"


namespace cnine{

  template<typename TYPE>
  class minivec{
  public:

    typedef std::size_t size_t;

    TYPE* arr;
    //mutable size_t memsize;
    size_t _size;
    int dev=0;

    ~minivec(){
      if(dev==0) delete[] arr;
      else CUDA_SAFE(cudaFree(arr));
    }


  public: //---- Constructors -------------------------------------


    minivec(const int n, const int _dev=0):
      _size(n), dev(_dev){
      if(_dev==0)
	arr=new TYPE[std::max(_size,(size_t)1)];
      else 
	CUDA_SAFE(cudaMalloc((void **)&arr,std::max(_size,(size_t)1)*sizeof(TYPE)));
    }

    minivec(const int n, const TYPE v, const int _dev):
      minivec(n,dev){
      std::fill(arr,arr+n,v);
      move_to_device(_dev);
    }


  public: //---- Copying -------------------------------------


    minivec(const minivec& x):
      minivec(x._size,x.dev){
      if(dev==0) std::copy(x.arr,x.arr+_size,arr);
      else 
	CUDA_SAFE(cudaMemcpy(arr,x.arr,_size*sizeof(TYPE),cudaMemcpyDeviceToDevice)); 
    }

    minivec(minivec&& x):
      _size(x._size){
      arr=x.arr;
      x.arr=nullptr;
    }

    minivec& operator=(const minivec& x)=delete;


  public: //---- Transport -----------------------------------


    minivec(const minivec& x, const int _dev):
      minivec(x._size,_dev){
      if(dev==0){
	if(x.dev==0) std::copy(x.arr,x.arr+_size,arr);
	if(x.dev==1) CUDA_SAFE(cudaMemcpy(arr,x.arr,_size*sizeof(TYPE),cudaMemcpyDeviceToHost));
      }
      if(dev==1){
	if(x.dev==0) CUDA_SAFE(cudaMemcpy(arr,x.arr,_size*sizeof(TYPE),cudaMemcpyHostToDevice));
	if(x.dev==1) CUDA_SAFE(cudaMemcpy(arr,x.arr,_size*sizeof(TYPE),cudaMemcpyDeviceToDevice));
      }
    }
    
    minivec& move_to_device(const int _dev){
      if(_dev==0 && dev==1){
	TYPE* t=new TYPE[std::max(_size,(size_t)1)];
	CUDA_SAFE(cudaMemcpy(t,arr,_size*sizeof(TYPE),cudaMemcpyDeviceToHost));
	CUDA_SAFE(cudaFree(arr));
	arr=t;
      }
      if(_dev==1 && dev==0){
	TYPE* t=nullptr;
	CUDA_SAFE(cudaMalloc((void **)&t,std::max(_size,(size_t)1)*sizeof(TYPE)));
	CUDA_SAFE(cudaMemcpy(t,arr,_size*sizeof(TYPE),cudaMemcpyHostToDevice));
	delete[] arr;
	arr=t;
      }
      dev=_dev;
      return *this;
    }


  public: //---- Access -------------------------------------


    int size() const{
      return _size;
    }
    
    TYPE operator[](const int i) const{
      return arr[i];
    }

    TYPE& operator[](const int i){
      return arr[i];
    }

    void set(const int i, const TYPE v){
      arr[i]=v;
    }

    void inc(const int i, const TYPE v){
      arr[i]+=v;
    }


  public: //---- I/O ---------------------------------------


    string str() const{
      ostringstream oss;
      oss<<"[";
      for(int i=0; i<_size; i++)
	oss<<arr[i]<<",";
      oss<<"\b]";
      return oss.str();
    }


    friend ostream& operator<<(ostream& stream, const minivec& v){
      stream<<v.str(); return stream;}

  };

}

#endif 
