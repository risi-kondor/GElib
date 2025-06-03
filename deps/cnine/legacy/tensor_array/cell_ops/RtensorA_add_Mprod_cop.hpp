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


#ifndef _RtensorA_add_Mprod_op
#define _RtensorA_add_Mprod_op

#include "GenericCop.hpp"

#ifdef _WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif 

#ifdef _WITH_CUBLAS
#include <cublas_v2.h>
extern cublasHandle_t cnine_cublas;
#endif 


namespace cnine{

  class RtensorA_add_Mprod_AA_cop: public BinaryCop<RtensorA,RtensorArrayA>{
  public:

    int nx,ny;

    RtensorA_add_Mprod_AA_cop(const int _nx, const int _ny): nx(_nx), ny(_ny){}
    
    virtual void operator()(RtensorA& r, const RtensorA& x, const RtensorA& y) const{
      assert(x.dev==r.dev);
      assert(y.dev==r.dev);

      if(x.asize==0 || y.asize==0) return;

      const int K=x.combined_size(x.k-nx,x.k);
      assert(y.combined_size(0,ny)==K);
      const int I=x.combined_size(0,x.k-nx);
      const int J=y.combined_size(ny,y.k);
      assert(r.asize==I*J);
      if(r.asize==0) return;

      if(r.dev==0){

	const int istridex=K;
	const int istrider=J;
	const int pstridey=J;

	for(int i=0; i<I; i++)
	  for(int j=0; j<J; j++){
	    float tr=0; 
	    for(int p=0; p<K; p++){
	      //cout<<i<<" "<<j<<" "<<p<<endl;
	      int qx=i*istridex+p;
	      int qy=p*pstridey+j;
	      float xr=x.arr[qx];
	      float yr=y.arr[qy];
	      tr+=xr*yr;
	    }
	    int qr=i*istrider+j;
	    r.arr[qr]+=tr;
	  }
      }    
    }

    template<typename IMAP>
    void operator()(const IMAP& map, RtensorArrayA& r, const RtensorArrayA& x, const RtensorArrayA& y) const{
      CNINE_UNIMPL();
    }

  };


    // The last nx indices of x are contracted with the last ny indices of y

  class RtensorA_add_Mprod_AT_cop: public BinaryCop<RtensorA,RtensorArrayA>{
  public:

    int nx,ny;

    RtensorA_add_Mprod_AT_cop(const int _nx, const int _ny): nx(_nx), ny(_ny){}
    
    virtual void operator()(RtensorA& r, const RtensorA& x, const RtensorA& y) const{
      assert(x.dev==r.dev);
      assert(y.dev==r.dev);

      if(x.asize==0 || y.asize==0) return;

      const int K=x.combined_size(x.k-nx,x.k);
      assert(y.combined_size(y.k-ny,y.k)==K);

      const int I=x.combined_size(0,x.k-nx);
      const int J=y.combined_size(0,y.k-ny);
      assert(r.asize==I*J);
      if(r.asize==0) return;

      if(r.dev==0){
	assert(x.dev==0);
	assert(y.dev==0);

	const int istridex=K;
	const int istrider=J;
	const int jstridey=K;

	for(int i=0; i<I; i++)
	  for(int j=0; j<J; j++){
	    float tr=0; 
	    for(int p=0; p<K; p++){
	      int qx=i*istridex+p;
	      int qy=p+j*jstridey;
	      float xr=x.arr[qx];
	      float yr=y.arr[qy];
	      tr+=xr*yr;
	    }
	    int qr=i*istrider+j;
	    r.arr[qr]+=tr;
	  }

      }

      if(r.dev>0){

	float alpha0=1.0;
	float beta=1.0;
	
	CUBLAS_SAFE(cublasSgemm(cnine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,J,I,K,&alpha0,
	    y.arrg,K,x.arrg,K,&beta,r.arrg,J)); 
	//cudaDeviceSynchronize(); 
      }
      
    }

    template<typename IMAP>
    void operator()(const IMAP& map, RtensorArrayA& r, const RtensorArrayA& x, const RtensorArrayA& y) const{
      CNINE_UNIMPL();
    }

  };


}

#endif
