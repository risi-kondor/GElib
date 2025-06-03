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

#ifndef _RtensorPool
#define _RtensorPool

#include "array_pool.hpp"
#include "vector_pool.hpp"
#include "RtensorA.hpp"
#include "Rtensor1_view.hpp"
#include "Rtensor2_view.hpp"
#include "Rtensor3_view.hpp"
#include "IntTensor.hpp"


namespace cnine{

  class RtensorPool{
  public:

    typedef cnine::Gdims Gdims;
    typedef cnine::RtensorA rtensor;
    typedef cnine::Rtensor1_view Rtensor1_view;
    typedef cnine::Rtensor2_view Rtensor2_view;
    typedef cnine::Rtensor3_view Rtensor3_view;

    float* arr=nullptr;
    float* arrg=nullptr;
    int dev=0;
    int memsize=0;
    int tail=0;
    vector_pool<int> headers;
    //IntTensor dir_mx;
    //bool dir_current=false;

    //bool is_view=false;

    ~RtensorPool(){
      //if(is_view) return;
      if(arr) delete[] arr;
      if(arrg) {CUDA_SAFE(cudaFree(arrg));}
    }


  public: // ---- Constructors -------------------------------------------------------------------------------


    RtensorPool(){}

    RtensorPool(const int _dev):
      dev(_dev){}

    RtensorPool(const int _N, const Gdims& _dims, const cnine::fill_raw& dummy, const int _dev=0):
      RtensorPool(_dev){
      int asize=_dims.asize();
      reserve(_N*asize);
      for(int i=0; i<_N; i++)
	headers.push_back_cat(i*asize,_dims);
      tail=_N*asize;
    }

    RtensorPool(const int _N, const Gdims& _dims, const cnine::fill_zero& dummy, const int _dev=0):
      RtensorPool(_dev){
      int asize=_dims.asize();
      reserve(_N*asize);
      if(dev==0) std::fill(arr,arr+memsize,0);
      if(dev==1){}
      for(int i=0; i<_N; i++)
	headers.push_back_cat(i*asize,_dims);
      tail=_N*asize;
    }

    RtensorPool(const int _N, const Gdims& _dims, const cnine::fill_gaussian& dummy, const int _dev=0):
      RtensorPool(_dev){
      CNINE_CPUONLY();
      int asize=_dims.asize();
      reserve(_N*asize);
      normal_distribution<double> distr;
      for(int i=0; i<_N*asize; i++) arr[i]=distr(rndGen);
      for(int i=0; i<_N; i++)
	headers.push_back_cat(i*asize,_dims);
      tail=_N*asize;
    }

    RtensorPool(const array_pool<int>& dimensions, const cnine::fill_zero& dummy, const int _dev=0){
      dev=_dev;

      int reserve_size=0;
      for(int i=0; i<dimensions.size(); i++){
	vector<int> v=dimensions(i);
	int t=1; for(int j=0; j<v.size(); j++) t*=v[j];
	reserve_size+=t;
      }
      reserve(reserve_size);
      if(dev==0) std::fill(arr,arr+reserve_size,0);
      if(dev==1){}

      for(int i=0; i<dimensions.size(); i++){
	vector<int> v=dimensions(i);
	int t=1; for(int j=0; j<v.size(); j++) t*=v[j];
	headers.push_back_cat(tail,v);
	tail+=t;
      }
    }

    RtensorPool(const rtensor& x){
      assert(x.ndims()==2);
      int m=x.dim(1);
      CNINE_CPUONLY();
      dev=x.dev;
      memsize=x.memsize;
      tail=memsize;
      if(x.dev==0){
	arr=new float[memsize];
	std::copy(x.arr,x.arr+memsize,arr);
      }
      for(int i=0; i<x.dim(0); i++)
	headers.push_back({i*m,m});
    }


  public: // ---- Named constructors -------------------------------------------------------------------------


    static RtensorPool zeros_like(const RtensorPool& x){
      RtensorPool R(x.dev);
      R.reserve(x.tail);
      if(x.dev==0) std::fill(R.arr,R.arr+x.tail,0);
      if(x.dev==1){}
      R.headers=x.headers;
      R.tail=x.tail;
      //R.dir_mx=x.dir_mx;
      //R.dir_current=x.dir_current;
      return R;
    }

    static RtensorPool* new_zeros_like(const RtensorPool& x){
      RtensorPool*  R=new RtensorPool(x.dev);
      R->reserve(x.tail);
      if(x.dev==0) std::fill(R->arr,R->arr+x.tail,0);
      if(x.dev==1){}
      R->headers=x.headers;
      R->tail=x.tail;
      //R.dir_mx=x.dir_mx;
      //R.dir_current=x.dir_current;
      return R;
    }


  public: // ---- Memory management --------------------------------------------------------------------------


    void reserve(const int n){
      if(n<=memsize) return;
      int newsize=std::max(n,1);
      if(dev==0){
	float* newarr=new float[newsize];
	if(arr){
	  std::copy(arr,arr+memsize,newarr);
	  delete[] arr;
	}
	arr=newarr;
	memsize=newsize;
      }
      if(dev==1){
	float* newarrg=nullptr;
	CUDA_SAFE(cudaMalloc((void **)&newarrg, newsize*sizeof(float)));
	if(arrg){
	  CUDA_SAFE(cudaMemcpy(newarrg,arrg,memsize*sizeof(float),cudaMemcpyDeviceToDevice));  
	  CUDA_SAFE(cudaFree(arrg));
	}
	arrg=newarrg;
	memsize=newsize;
      }
    }


    void reserve_zero(const int n){
      if(n<=memsize) return;
      //int newsize=n;
      if(dev==0){
	float* newarr=new float[n];
	if(arr){
	  std::copy(arr,arr+memsize,newarr);
	  delete[] arr;
	}
	arr=newarr;
	std::fill(arr+memsize,arr+n,0);
	memsize=n;
      }
      if(dev==1){
	float* newarrg=nullptr;
	CUDA_SAFE(cudaMalloc((void **)&newarrg, n*sizeof(float)));
	if(arrg){
	  CUDA_SAFE(cudaMemcpy(newarrg,arrg,memsize*sizeof(float),cudaMemcpyDeviceToDevice));  
	  CUDA_SAFE(cudaFree(arrg));
	}
	arrg=newarrg;
	CUDA_SAFE(cudaMemset(arrg+memsize,0,(n-memsize)*sizeof(float)));
	memsize=n;
      }
    }


  public: // ---- Copying ------------------------------------------------------------------------------------


    RtensorPool(const RtensorPool& x){
      CNINE_COPY_WARNING();
      dev=x.dev;
      tail=x.tail;
      memsize=tail;
      headers=x.headers;
      if(dev==0){
	arr=new float[memsize];
	std::copy(x.arr,x.arr+memsize,arr);
      }
      if(dev==1){
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
	CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(float),cudaMemcpyDeviceToDevice));  
      }
      //dir_mx=x.dir_mx;
      //dir_current=x.dir_current;
    }

    RtensorPool(RtensorPool&& x){
      CNINE_MOVE_WARNING();
      dev=x.dev;
      tail=x.tail; x.tail=0;
      memsize=x.memsize; x.memsize=0; 
      arr=x.arr; x.arr=nullptr;
      arrg=x.arrg; x.arrg=nullptr;
      headers=std::move(x.headers);
    }

    RtensorPool& operator=(const RtensorPool& x)=delete;


  public: // ---- Access -------------------------------------------------------------------------------------


    int get_dev() const{
      return dev;
    }

    int size() const{
      return headers.size();
    }

    float* get_arr() const{
      if(dev==0) return arr;
      else return arrg;
    }


    int addr_of(const int i) const{
      CNINE_IN_RANGE(i,size());
      return headers(i)[0];
    }

    cnine::Gdims dims_of(const int i) const{
      CNINE_IN_RANGE(i,size());
      return cnine::Gdims(headers.subvector_of(i,1));
    }

    int dim_of(const int i, const int j) const{
      CNINE_IN_RANGE(i,size());
      return headers(i,1+j);
    }

    float* arr_of(const int i) const{
      if(dev==1) return arrg+addr_of(i);
      return arr+addr_of(i);
    }


    rtensor operator()(const int i) const{
      CNINE_IN_RANGE(i,size());
      return rtensor(dims_of(i),get_arr()+addr_of(i),dev);
    }

    rtensor view_of_tensor(const int i){
      CNINE_IN_RANGE(i,size());
      return rtensor::view_of_blob(dims_of(i),get_arr()+addr_of(i),dev);
    }

    const rtensor view_of_tensor(const int i) const{
      CNINE_IN_RANGE(i,size());
      return rtensor::view_of_blob(dims_of(i),get_arr()+addr_of(i),dev);
    }

    //rtensor tensor(const int i) const{
    //assert(i<size());
    //return rtensor(dims_of(i),get_arr()+addr_of(i),dev);
    //}

    Rtensor1_view view1_of(const int i) const{
      CNINE_IN_RANGE(i,size());
      vector<int> v=headers(i);
      assert(v.size()==2);
      if(dev==1) return Rtensor1_view(arrg+v[0],v[1],1,1);
      return Rtensor1_view(arr+v[0],v[1],1,0);
    }

    Rtensor2_view view2_of(const int i) const{
      CNINE_IN_RANGE(i,size());
      vector<int> v=headers(i);
      assert(v.size()==3);
      if(dev==1) return Rtensor2_view(arrg+v[0],v[1],v[2],v[2],1,1);
      return Rtensor2_view(arr+v[0],v[1],v[2],v[2],1,0);
    }

    Rtensor3_view view3_of(const int i) const{
      CNINE_IN_RANGE(i,size());
      vector<int> v=headers(i);
      assert(v.size()==4);
      if(dev==1) return Rtensor3_view(arrg+v[0],v[1],v[2],v[3],v[2]*v[3],v[3],1,1);
      return Rtensor3_view(arr+v[0],v[1],v[2],v[3],v[2]*v[3],v[3],1,0);
    }



    void push_back(const rtensor& x){
      assert(x.dev==dev);
      if(tail+x.asize>memsize)
	reserve(std::max(2*memsize,tail+x.asize));
      if(dev==0){
	std::copy(x.arr,x.arr+x.asize,arr+tail);
      }
      if(dev==1){
	CUDA_SAFE(cudaMemcpy(arrg+tail,x.arrg,x.asize*sizeof(float),cudaMemcpyDeviceToDevice));
      }
      headers.push_back_cat(tail,x.dims);
      tail+=x.asize;
    }

    void push_back_raw(const Gdims& _dims){
      int asize=_dims.asize();
      if(tail+asize>memsize)
	reserve(std::max(2*memsize,tail+asize));
      headers.push_back_cat(tail,_dims);
      tail+=asize;
    }
      
    void push_back_zero(const Gdims& _dims){
      int asize=_dims.asize();
      if(tail+asize>memsize)
	reserve(std::max(2*memsize,tail+asize));
      if(dev==0){
	std::fill(arr+tail,arr+tail+asize,0);
      }
      if(dev==1){
	CUDA_SAFE(cudaMemset(arrg+tail,0,asize*sizeof(float)));
      }
      headers.push_back_cat(tail,_dims);
      tail+=asize;
    }

  public: // ---- Cumulative operations ----------------------------------------------------------------------


    void add(const RtensorPool& x){
      assert(x.dev==dev);
      assert(x.tail==tail);
      CPUCODE(cnine::stdadd(x.arr,x.arr+tail,arr));
      GPUCODE(const float alpha = 1.0; CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &alpha, x.arrg, 1, arrg, 1)));
    }


    void add(const RtensorPool& x, const float c){
      assert(x.dev==dev);
      assert(x.tail==tail);
      CPUCODE(cnine::stdadd(x.arr,x.arr+tail,arr,c));
      GPUCODE(const float alpha = c; CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &alpha, x.arrg, 1, arrg, 1)));
    }


    void add_ReLU(const RtensorPool& x, const float alpha=0.1){
      CNINE_CPUONLY();
      assert(x.dev==dev);
      assert(x.tail==tail);
      CPUCODE(for(int i=0; i<tail; i++) if(x.arr[i]>0) arr[i]+=x.arr[i]; else arr[i]+=alpha*x.arr[i]);
      GPUCODE();
    }

    void add_ReLU_back(const cnine::RtensorPool& x, const float alpha=0.1){
      CNINE_CPUONLY();
      assert(x.dev==dev);
      assert(x.tail==tail);
      CPUCODE(for(int i=0; i<tail; i++) if(x.arr[i]>0) arr[i]=x.arr[i]; else arr[i]=x.arr[i]*alpha);
      GPUCODE();
    }


  public: // ---- Operations ---------------------------------------------------------------------------------

    
    float inp(const RtensorPool& y){
      CNINE_ASSRT(tail==y.tail);
      float t=0;
      CNINE_CPUONLY();
      CPUCODE(for(int i=0; i<tail; i++) t+=arr[i]*y.arr[i];)
      return t;
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "RtensorPool";
    }

    string str(const string indent="") const{
      ostringstream oss;
      for(int i=0; i<size(); i++){
	oss<<indent<<"Tensor "<<i<<":"<<endl;
	oss<<(*this)(i).str(indent)<<endl;
      }
      return oss.str();
    }


    friend ostream& operator<<(ostream& stream, const RtensorPool& v){
      stream<<v.str(); return stream;}

  };

}

#endif
