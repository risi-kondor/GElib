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


#ifndef _Flock
#define _Flock

#include "Cnine_base.hpp"


namespace cnine{


  template<typename OBJ>
  class Flock{
  public:

    int N;
    int memsize; 
    int dev=0;
    //bool is_view=true; 
    float* arr=nullptr; 
    float** arr_array=nullptr;
    
    vector<OBJ*> obj;

    Flock(){}

    ~Flock(){
      if(dev==0 && arr_array) delete[] arr_array;
      if(dev==1 && arr_array) CUDA_SAFE(cudaFree(arr_array));
      if(dev==0 && arr) delete[] arr;
      if(dev==1 && arr) CUDA_SAFE(cudaFree(arr));
    }
    

  public: // ---- Constructors -------------------------------------------------------------------------------


    Flock(const vector<OBJ*>& v):
      N(v.size()), obj(v){
      if(N==0) return;
      memsize=obj[0]->memsize;
      dev=obj[0]->dev;
      
      if(dev==0){
	arr_array=new float*[N];
	for(int i=0; i<N; i++)
	  arr_array[i]=v[i]->arr;
      }

      if(dev==1){
	float* buffer[N];
	for(int i=0; i<N; i++){
	  buffer[i]=v[i]->arr;
	}
	CUDA_SAFE(cudaMalloc((void ***)&arr_array, N*sizeof(float*)));
	CUDA_SAFE(cudaMemcpy(arr_array,buffer,N*sizeof(float*),cudaMemcpyHostToDevice));  
      }
    }


    Flock(const OBJ& model, const int _N){ // use as buffer
      N=_N; 
      memsize=model.memsize;
      dev=model.dev;
      
      if(dev==0){
	arr=new float[N*memsize];
	arr_array=new float*[N];
	for(int i=0; i<N; i++)
	  arr_array[i]=arr+i*memsize;
      }

      if(dev==1){
	CUDA_SAFE(cudaMalloc((void **)&arr, N*memsize*sizeof(float)));
	float* buffer[N];
	for(int i=0; i<N; i++){
	  buffer[i]=arr+i*memsize;
	}
	CUDA_SAFE(cudaMalloc((void ***)&arr_array, N*sizeof(float*)));
	CUDA_SAFE(cudaMemcpy(arr_array,buffer,N*sizeof(float*),cudaMemcpyHostToDevice));  
      }

    }

    
  public: // ---- Cumulative Operations ----------------------------------------------------------------------


#ifdef _WITH_CUDA
    void copy_cu(const Flock& x, const cudaStream_t& stream);
    void add_cu(const Flock& x, const cudaStream_t& stream);
    void sum_into_cu(const OBJ& R, const cudaStream_t& stream);
#endif 

    
    void copy(const Flock& x){
      assert(x.dev==dev);
      assert(x.N==N);
      assert(x.memsize==memsize);

      if(dev==0){
	for(int i=0; i<N; i++)
	  std::copy(x.arr_array[i],x.arr_array[i]+memsize,arr_array[i]);
      }else{
#ifdef _WITH_CUDA
	x.to_device(1);
	cudaStream_t stream;
	CUDA_SAFE(cudaStreamCreate(&stream));
	copy_cu(x,stream);
	CUDA_SAFE(cudaStreamSynchronize(stream));
	CUDA_SAFE(cudaStreamDestroy(stream));
#else
	CNINE_NOCUDA_ERROR;
#endif
      }
    }
    

    void add(const Flock& x){
      assert(x.dev==dev);
      assert(x.N==N);
      assert(x.memsize==memsize);
      
      if(dev==0){
	for(int i=0; i<N; i++)
	  fastadd(x.arr_array[i],arr_array[i],memsize);
      }else{
#ifdef _WITH_CUDA
	cudaStream_t stream;
	CUDA_SAFE(cudaStreamCreate(&stream));
	add_cu(x,stream);
	CUDA_SAFE(cudaStreamSynchronize(stream));
	CUDA_SAFE(cudaStreamDestroy(stream));
	//CUBLAS_SAFE(gemmBatched(genet_cublas,CUBLAS_OP_N,CUBLAS_OP_N);)
#endif
      }
    }


    void sum_into(OBJ& R){
      assert(R.dev==dev);
      assert(R.memsize==memsize);

      if(dev==0){
	for(int i=0; i<N; i++)
	  fastadd(arr_array[i],R.arr,memsize);
      }else{
#ifdef _WITH_CUDA
	cudaStream_t stream;
	CUDA_SAFE(cudaStreamCreate(&stream));
	sum_into_cu(R,stream);
	CUDA_SAFE(cudaStreamSynchronize(stream));
	CUDA_SAFE(cudaStreamDestroy(stream));
#else
	CNINE_NOCUDA_ERROR;
#endif
      }

    }
    
  };


  /*
  template<typename OBJ>
  class FlockBuffer: public Flock<OBJ>{
  public:

    float* arr=nullptr; 

    ~FlockBuffer(){
      if(dev==0) delete[] arr;
      if(dev==1) CUDA_SAFE(cudaFree(arr));
    }

  public:

    FlockBuffer(const OBJ& model, const int _N){
      N=_N; 
      memsize=model.memsize;
      dev=model.dev;
      
      if(dev==0){
	arr=new float[N*memsize];
	arr_array=new float*[N];
	for(int i=0; i<N; i++)
	  arr_array[i]=arr+i*memsize;
      }

      if(dev==1){
	CUDA_SAFE(cudaMalloc((void **)&arr, N*memsize*sizeof(float)));
	float* buffer[N];
	for(int i=0; i<N; i++){
	  buffer[i]=arr+i*memsize;
	}
	CUDA_SAFE(cudaMalloc((void ***)&arr_array, N*sizeof(float*)));
	CUDA_SAFE(cudaMemcpy(arr_array,buffer,N*sizeof(float*),cudaMemcpyHostToDevice));  
      }

    }
    
  };
  */

}


#endif 
