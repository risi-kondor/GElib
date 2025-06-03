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


#ifndef _CnineSession
#define _CnineSession

#include "Cnine_base.hpp"
#include "CnineLog.hpp"
#include <chrono>
#include <ctime>


#ifdef _WITH_CENGINE
#include "CengineSession.hpp"
#endif 

#ifdef _WITH_CUBLAS
#include <cublas_v2.h>
extern cublasHandle_t cnine_cublas;
#endif

extern cnine::CnineLog cnine_log;
//extern cnine::CallStack call_stack;


namespace cnine{

  extern int streaming_footprint;
  extern float* cuda_oneS;

  extern thread_local int nthreads;


  class cnine_session{
  public:

    std::time_t start_time;


    #ifdef _WITH_CENGINE
    Cengine::CengineSession* cengine_session=nullptr;
    #endif


    cnine_session(const int _nthreads=1){

#ifdef _WITH_CUDA
      cout<<"Starting cnine tensor library with CUDA support..."<<endl;
#else
      cout<<"Starting cnine tensor library without CUDA support..."<<endl;
#endif

      /*
      cout<<"-------------------------------------"<<endl;
      cout<<"Cnine tensor library  (c) Risi Kondor"<<endl;

#ifdef _WITH_CUDA
      cout<<"CUDA support:                      ON"<<endl;
#else
      cout<<"CUDA support:                     OFF"<<endl;
#endif

#ifdef CNINE_RANGE_CHECKING
      cout<<"Range checking:                    ON"<<endl;
#else
      cout<<"Range checking:                   OFF"<<endl;
#endif

      cout<<"Number of CPU threads:              "<<nthreads<<endl;
      cout<<"-------------------------------------"<<endl;
      */

      start_time=std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

      nthreads=_nthreads;

      #ifdef _WITH_CENGINE
      cengine_session=new Cengine::CengineSession();
      #endif

      #ifdef _WITH_CUDA
      float a=1.0;
      CUDA_SAFE(cudaMalloc((void **)&cuda_oneS, sizeof(float)));
      CUDA_SAFE(cudaMemcpy(cuda_oneS,&a,sizeof(float),cudaMemcpyHostToDevice)); 
      #endif 

      #ifdef _WITH_CUBLAS
      cublasCreate(&cnine_cublas);
      #endif 

    }


    ~cnine_session(){
#ifdef _WITH_CENGINE
      delete cengine_session;
#endif 
    }
    

  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      cout<<indent<<"cnine session started "<<std::ctime(&start_time);
      cout<<indent<<"Number of CPU threads: "<<nthreads<<endl;
      cout<<indent<<"GPU footprint for streaming operations: "<<streaming_footprint<<" MB"<<endl;
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const cnine_session& x){
      stream<<x.str(); return stream;
    }

  };

}


#endif 
