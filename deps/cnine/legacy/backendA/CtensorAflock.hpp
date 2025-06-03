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


#ifndef _CtensorAflock
#define _CtensorAflock

#include "CtensorA.hpp"
#include "Flock.hpp"
#include "Cnode.hpp" // not ideal, should template out 

namespace cnine{

  template<typename TYPE>
  vector<TYPE*> flockSelector(const vector<Cnode*>& v, const int s){
    const int N=v.size();
    vector<TYPE*> R(N);
    for(int i=0; i<N; i++)
      R[i]=dynamic_cast<TYPE*>(v[i]->op->inputs[s]->obj);
    return R;
  }
    

  class CtensorAflock: public Flock<CtensorA>{
  public:

    Gdims dims; 
    //int nbu=-1;


  public:

    CtensorAflock(const vector<CtensorA*>& v):
      Flock(v){
      if(N==0) return;
      dims=v[0]->dims;
      //nbu=v[0]->nbu;
    }

    CtensorAflock(const vector<Cnode*> v, const int s):
      Flock(flockSelector<CtensorA>(v,s)){
      if(N==0) return;
      CtensorA& model=*dynamic_cast<CtensorA*>(v[0]->op->inputs[s]->obj);
      dims=model.dims;
      //nbu=v[0]->nbu;
    }


    CtensorAflock(const CtensorA& model, const int _N):
      Flock(model,_N){
      dims=model.dims;
    }


  public: // -------------------------------------------------------------------------------------------------


  public: // ---- Cumulative Operations ----------------------------------------------------------------------



    //void add_inp_into(CscalarBpack& R, CtensorA_flock& Y){
    //CENGINE_UNIMPL(); 
    //}
    
    // #include "CtensorA_flock_add_Mprod.hpp"


  };


}

#endif


    /*
    CtensorA_flock(){}

    CtensorA_flock(const int _N, const Gdims& _dims, const int _nbu=-1, const int dev=1):
      N(_N), dims(_dims), nbu(_nbu), device(dev){
      CUDA_SAFE(cudaMalloc((void ***)&parr, N*sizeof(float*)));
      CUDA_SAFE(cudaMalloc((void ***)&parrc, N*sizeof(float*)));
      parr_valid=true;
    }

    CtensorA_flock(const int _N, const CtensorA& x):
      CtensorA_flock(_N,x.dims,x.nbu,x.device){
      memsize=x.memsize;
    }

    CtensorA_flock(const CtensorB& x):
      dims(x.dims), nbu(x.nbu), device(x.device), memsize(x.memsize){}
    
    CtensorA_flock(const vector<CtensorA*>& v):
      CtensorA_flock(v.size(),*v[0]){
      memsize=v[0]->memsize;
      pack.resize(N);
      for(int i=0; i<N; i++){
	pack[i]=v[i];
	pack[i]->to_device(device);
      }
      float* arr[N]; 
      float* arrc[N]; 
      for(int i=0; i<N; i++){
	arr[i]=pack[i]->arrg;
	arrc[i]=pack[i]->arrgc;
      }
      CUDA_SAFE(cudaMemcpy(parr,arr,N*sizeof(float*),cudaMemcpyHostToDevice));  
      CUDA_SAFE(cudaMemcpy(parrc,arrc,N*sizeof(float*),cudaMemcpyHostToDevice));  
    }

    CtensorA_flock(const vector<Cnode*>& v, const int s):
      CtensorA_flock(v.size(),*dynamic_cast<CtensorB*>(v[0]->op->inputs[s]->obj)){
      pack.resize(N);
      for(int i=0; i<N; i++){
	pack[i]=dynamic_cast<CtensorB*>(v[i]->op->inputs[s]->obj);
	pack[i]->to_device(device);
      }
      memsize=pack[0]->memsize;
      float* arr[N]; 
      float* arrc[N]; 
      for(int i=0; i<N; i++){
	arr[i]=pack[i]->arrg;
	arrc[i]=pack[i]->arrgc;
      }
      CUDA_SAFE(cudaMemcpy(parr,arr,N*sizeof(float*),cudaMemcpyHostToDevice));  
      CUDA_SAFE(cudaMemcpy(parrc,arrc,N*sizeof(float*),cudaMemcpyHostToDevice));  
    }

    ~CtensorA_flock(){
      if(parr) CUDA_SAFE(cudaFree(parr));
      if(parrc) CUDA_SAFE(cudaFree(parrc));
    }

    CtensorA_flock(const CtensorA_flock& x)=delete;
    CtensorA_flock& operator=(const CtensorA_flock& x)=delete;

    CtensorA_flock& operator=(CtensorA_flock&& x){
      if(parr) CUDA_SAFE(cudaFree(parr));
      if(parrc) CUDA_SAFE(cudaFree(parrc));
      N=x.N; dims=x.dims; nbu=x.nbu; device=x.device; 
      parr=x.parr; x.parr=nullptr;
      parrc=x.parrc; x.parrc=nullptr;
      pack=std::move(x.pack);
      memsize=x.memsize;
      return *this; 
    }
    */
    /*
    int get_nbu() const{
      return nbu;
    }

    Gdims get_dims() const{
      return dims; 
    }


    float** get_parr() const{
      if(!parr || !parr_valid) renew_parr();
      return parr;
    }


    float** get_parrc() const{
      if(!parrc || !parr_valid) renew_parr();
      return parrc;
    }


    void renew_parr() const{
    if(parr) CUDA_SAFE(cudaFree(parr)); parr=nullptr;
      if(parrc) CUDA_SAFE(cudaFree(parrc)); parrc=nullptr;

      to_device(1);
      const int N=pack.size(); 
      float* arr[N]; 
      float* arrc[N]; 
      for(int i=0; i<N; i++){
	arr[i]=pack[i]->arrg;
	arrc[i]=pack[i]->arrgc;
      }
      
      CUDA_SAFE(cudaMalloc((void ***)&parr, N*sizeof(float*)));
      CUDA_SAFE(cudaMemcpy(parr,arr,N*sizeof(float*),cudaMemcpyHostToDevice));  
      CUDA_SAFE(cudaMalloc((void ***)&parrc, N*sizeof(float*)));
      CUDA_SAFE(cudaMemcpy(parrc,arrc,N*sizeof(float*),cudaMemcpyHostToDevice));  

      device=1;
      parr_valid=true;
    }


    void to_device(const device_id& _dev) const{
      assert(false); 
      if(_dev.id()==device) return; 
      parr_valid=false; 
      if(parr) CUDA_SAFE(cudaFree(parr)); parr=nullptr;
      if(parrc) CUDA_SAFE(cudaFree(parrc)); parrc=nullptr;
      device=_dev.id();
      for(auto p: pack)
	p->to_device(device);
    }
    */


