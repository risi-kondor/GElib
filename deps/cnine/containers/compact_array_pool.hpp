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

#ifndef _compact_array_pool
#define _compact_array_pool

#include "Cnine_base.hpp"
#include "array_pool.hpp"

namespace cnine{

  template<typename TYPE>
  class compact_array_pool{
  public:

    int n;
    int last=-1;
    int memsize;
    int dev=0;
    TYPE* arr=nullptr;

    ~compact_array_pool(){
      if(dev==0 && arr) delete[] arr;
      if(dev>0 && arr) CUDA_SAFE(cudaFree(arr));
    }
    
    compact_array_pool(const int _n, const int _m):
      n(_n){
      memsize=_n+_m+2;
      arr=new TYPE[memsize];
      arr[0]=n;
      arr[1]=n+2; 
    }
    

  public: // ---- Copying ------------------------------------------------------------------------------------


    compact_array_pool(const compact_array_pool& x):
      n(x.n), 
      last(x.last), 
      memsize(x.memsize), 
      dev(x.dev){
      if(dev==0){
	arr=new TYPE[memsize];
	std::copy(x.arr,x.arr+memsize,arr);
      }
      if(dev==1){
	CUDA_SAFE(cudaMalloc((void **)&arr, std::max(memsize,1)*sizeof(TYPE)));
	CUDA_SAFE(cudaMemcpy(arr,x.arr,x.memsize*sizeof(TYPE),cudaMemcpyDeviceToDevice));  
      }
    }

    compact_array_pool(compact_array_pool&& x):
      n(x.n), 
      last(x.last), 
      memsize(x.memsize), 
      dev(x.dev){
      arr=x.arr;
      x.arr=nullptr;
    }

    compact_array_pool& operator=(const compact_array_pool& x){
      if(dev==0 && arr) delete[] arr;
      if(dev>0 && arr) CUDA_SAFE(cudaFree(arr));
      n=x.n;
      last=x.last;
      memsize=x.memsize;
      dev=x.dev;
      if(dev==0){
	arr=new TYPE[memsize];
	std::copy(x.arr,x.arr+memsize,arr);
      }
      if(dev==1){
	CUDA_SAFE(cudaMalloc((void **)&arr, std::max(memsize,1)*sizeof(TYPE)));
	CUDA_SAFE(cudaMemcpy(arr,x.arr,x.memsize*sizeof(TYPE),cudaMemcpyDeviceToDevice));  
      }
      return *this;
    };


  public: // ---- Transport ----------------------------------------------------------------------------------


    compact_array_pool(const compact_array_pool& x, const int _dev):
      n(x.n), 
      last(x.last), 
      memsize(x.memsize), 
      dev(x.dev){
      if(dev==x.dev) return compact_array_pool(x);
      if(dev==0){
	arr=new TYPE[memsize];
	CUDA_SAFE(cudaMemcpy(arr,x.arr,x.memsize*sizeof(TYPE),cudaMemcpyDeviceToHost));  
      }
      if(dev==1){
	CUDA_SAFE(cudaMalloc((void **)&arr, std::max(memsize,1)*sizeof(TYPE)));
	CUDA_SAFE(cudaMemcpy(arr,x.arr,x.memsize*sizeof(TYPE),cudaMemcpyHostToDevice));  
      }
    }

    compact_array_pool& to_device(const int _dev){
      if(_dev==dev) return *this;
      if(_dev==0){
	narr=new TYPE[memsize];
	CUDA_SAFE(cudaMemcpy(narr,arr,memsize*sizeof(TYPE),cudaMemcpyDeviceToHost));  
	CUDA_SAFE(cudaFree(arr));
	arr=narr;
	dev=_dev;
      }
     if(_dev>0){
	TYPE* narr;
	CUDA_SAFE(cudaMalloc((void **)&narr, std::max(memsize,1)*sizeof(TYPE)));
	CUDA_SAFE(cudaMemcpy(narr,arr,memsize*sizeof(TYPE),cudaMemcpyHostToDevice));  
	delete[] arr;
	arr=narr;
	dev=_dev;
      }
    }


  public: // ---- Getters ------------------------------------------------------------------------------------


    int get_dev() const{
      return dev;
    }

    int size() const{
      return n;
    }

    int tail() const{
      CNINE_CPUONLY();
      return arr[last+2];
    }

    int offset(const int i) const{
      CNINE_IN_RANGE(i,n);
      CNINE_CPUONLY();
      return arr[i+1];
    }

    int size_of(const int i) const{
      CNINE_IN_RANGE(i,n);
      CNINE_CPUONLY();
      return arr[i+2]-arr[i+1];
    }

    int operator()(const int i, const int j) const{
      CNINE_IN_RANGE(i,n);
      CNINE_IN_RANGE(j,size_of(i));
      CNINE_CPUONLY();
      return arr[arr[i+1]+j];
    }


  public: // ---- Setters ------------------------------------------------------------------------------------


    void set(const int i, const int j, const int v){
      CNINE_IN_RANGE(i,n);
      CNINE_IN_RANGE(j,size_of(i));
      CNINE_CPUONLY();
      arr[arr[i+1]+j]=v;
    }


  public: // ---- Conversions ---------------------------------------------------------------------------------


    compact_array_pool(const array_pool<TYPE>& x, const int _dev=0){
      if(x.dev>0){
	*this=compact_array_pool(array_pool<TYPE>(x,0),_dev);
	return;
      }
      n=x.size();
      last=x.size();
      memsize=x.size()+x.tail+2;
      arr=new TYPE[memsize];
      arr[0]=n;
      arr[1]=0;
      int t=n+2;
      for(int i=0; i<n; i++){
	std::copy(x.arr+x.offset(i)+x.arr+x.offset(i)+x.size_of(i),arr+t);
	t+=x.size_of(i);
	arr[i+2]=t;
      }
      move_to_device(_dev);
    }


  public: // ---- I/O -----------------------------------------------------------------------------------------


    string classname() const{
      return "compact_array_pool";
    }

    string str(const string indent="") const{
      if(dev>0) return compact_array_pool(*this,0);
      ostringstream oss;
      for(int i=0; i<n; i++){
	oss<<indent<<i<<":(";
	for(int j=0; j<size_of(i); j++)
	  oss<<(*this)(i,j)<<",";
	if(size_of(i)>0) oss<<"\b";
	oss<<")"<<endl;
      //if(is_labeled) oss<<labels.str(indent+"  ")<<endl;
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const compact_array_pool& x){
      stream<<x.str(); return stream;}

  };

};

#endif 

