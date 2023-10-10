// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3part_addCGproductFn
#define _SO3part_addCGproductFn

#include "GElib_base.hpp"
#include "CtensorB.hpp"
#include "SO3part2_view.hpp"
#include "SO3part4_view.hpp"
#include "MultiLoop.hpp"
#include "GElibTimer.hpp"
#include "WorkStreamLoop.hpp"

extern thread_local cnine::DeviceSelector cnine::dev_selector;

extern GElib::SO3_CGbank SO3_cgbank;
extern GElib::SO3_SPHgen SO3_sphGen;


namespace GElib{

  #ifdef _WITH_CUDA
  void SO3partB_addCGproduct_cu(cnine::Ctensor3_view r, const cnine::Ctensor3_view& x, const cnine::Ctensor3_view& y, 
    const int offs, const cudaStream_t& stream);
  void SO3partB_addDiagCGproduct_cu(cnine::Ctensor3_view r, const cnine::Ctensor3_view& x, const cnine::Ctensor3_view& y, 
    const int offs, const cudaStream_t& stream);
  #endif


  class SO3part_addCGproductFn{
  public:

    void operator()(const SO3part3_view& _r, const SO3part3_view& _x, const SO3part3_view& _y, const int _offs=0){


      const int l=_r.getl(); 
      const int l1=_x.getl(); 
      const int l2=_y.getl();
 
      const int N1=_x.n2;
      const int N2=_y.n2;
      const int B=_x.n0;
      const int dev=_r.dev;

      CNINE_CHECK_DEV3(_r,_x,_y);
      CNINE_CHECK_BATCH3(_r,_x,_y);

      assert(_offs+N1*N2<=_r.n2);
      assert(l>=abs(l1-l2) && l<=l1+l2);

      int count=0; for(int i=-l1; i<=l1; i++) count+=std::min(l2,l-i)-std::max(-l2,-l-i)+(i<=l);
      CGproductTimer timer(l1,l2,l,B,N1,N2,dev,B*count*N1*N2);
      
      if(dev==0 && cnine::dev_selector.dev>0){
	int sdev=cnine::dev_selector.dev;

	if(!_r.is_regular() || !_x.is_regular() || !_y.is_regular()){
	  gelib_log->error(__PRETTY_FUNCTION__,"Arguments of streaming operation must have regular strides. Skipping this operation."); return;}

	int nb=std::min((cnine::dev_selector.max_mem<<18)/(2*_x.n1*_x.n2+2*_y.n1*_y.n2+2*_r.n1*_x.n2*_y.n2),_r.n0);
	int rsize_per_batch=_r.n1*_x.n2*_y.n2;
	//cout<<"nb="<<nb<<endl;
	cnine::ArrayOnDevice<float> xbuf(2*nb*_x.n1*_x.n2);
	cnine::ArrayOnDevice<float> ybuf(2*nb*_y.n1*_y.n2);
	cnine::ArrayOnDevice<float> rbuf(2*nb*rsize_per_batch);
	cnine::Ctensor3_view xv(xbuf,nb,_x.n1,_x.n2,_x.s0,_x.s1,_x.s2,1,sdev);
	cnine::Ctensor3_view yv(ybuf,nb,_y.n1,_y.n2,_y.s0,_y.s1,_y.s2,1,sdev);
	cnine::Ctensor3_view rv(rbuf,nb,_r.n1,_x.n2*_y.n2,_r.s0,_r.s1,_r.s2,1,sdev);

#ifdef _WITH_CUDA
	cu_stream stream;
	//cudaStream_t stream;
	//CUDA_SAFE(cudaStreamCreate(&stream));
	cout<<"Streaming packets: "<<cnine::roundup(_r.n0,nb)/nb<<" "<<nb<<endl;
	for(int i=0; i<cnine::roundup(_r.n0,nb)/nb; i++){
	  cout<<i<<" "<<i*nb*_r.s0+2*_offs<<endl;
	  int _nb=std::min(nb,_x.n0-i*nb);
	  if(nb==0) continue;
	  if(_nb<nb){
	    xv.n0=_nb;
	    yv.n0=_nb;
	    rv.n0=_nb;
	  }

	  CUDA_SAFE(cudaMemcpyAsync(xbuf, _x.arr+i*nb*_x.s0, _nb*_x.s0*sizeof(float), cudaMemcpyHostToDevice, stream));
	  CUDA_SAFE(cudaMemcpyAsync(ybuf, _y.arr+i*nb*_y.s0, _nb*_y.s0*sizeof(float), cudaMemcpyHostToDevice, stream));
	  //CUDA_SAFE(cudaMemcpyAsync(rbuf, _r.arr+i*nb*_r.s0+2*_offs, 2*_nb*rsize_per_batch*sizeof(float), 
	  //    cudaMemcpyHostToDevice, stream));
	  SO3partB_addCGproduct_cu(rv,xv,yv,0,stream);
	  //CUDA_SAFE(cudaMemcpyAsync(_r.arr+i*nb*_r.s0+2*_offs, rbuf, 2*_nb*rsize_per_batch*sizeof(float), 
	  //  cudaMemcpyDeviceToHost, stream));

	}
	//CUDA_SAFE(cudaStreamSynchronize(stream));
	//CUDA_SAFE(cudaStreamDestroy(stream));
#endif
	return;
      }

      if(dev==0){

	auto& C=SO3_cgbank.getf(CGindex(l1,l2,l));
	cnine::MultiLoop(B,[&](const int b){
	    SO3part2_view r=_r.slice0(b);
	    SO3part2_view x=_x.slice0(b);
	    SO3part2_view y=_y.slice0(b);
	    int offs=_offs;
	    
	    for(int n1=0; n1<N1; n1++){
	      for(int n2=0; n2<N2; n2++){
		for(int m1=-l1; m1<=l1; m1++){
		  for(int m2=std::max(-l2,-l-m1); m2<=std::min(l2,l-m1); m2++){
		    r.inc(m1+m2,offs+n2,C(m1+l1,m2+l2)*x(m1,n1)*y(m2,n2));
		  }
		}
	      }
	      offs+=N2;
	    }
	  });
      }

      if(dev==1) CUDA_STREAM(SO3partB_addCGproduct_cu(_r,_x,_y,_offs,stream));
      
    }


    void operator()(const cnine::Ctensor4_view& _r, const cnine::Ctensor5_view& _x, const cnine::Ctensor4_view& _y, const int _offs=0){

      const int l=(_r.n2-1)/2;
      const int l1=(_x.n3-1)/2;
      const int l2=(_r.n2-1)/2;
 
      const int N1=_x.n4;
      const int N2=_y.n3;
      const int B=_x.n0;
      const int J=_x.n1;
      const int K=_x.n2;
      const int dev=_r.dev;

      CNINE_CHECK_DEV3(_r,_x,_y);
      CNINE_CHECK_BATCH3(_r,_x,_y);
      CNINE_ASSRT(_x.n1==_r.n1);
      CNINE_ASSRT(_x.n2==_y.n1);
      CNINE_ASSRT(_offs+N1*N2<=_r.n3);
      CNINE_ASSRT(l>=abs(l1-l2) && l<=l1+l2);

      LoggedTimer timer("  ReducingCGproduct("+to_string(l1)+","+to_string(l2)+","+to_string(l)+")[b="+
	to_string(B)+", K="+to_string(K)+",n1="+to_string(N1)+",n2="+to_string(N2)+",dev="+to_string(dev)+"]",B*K*(2*l1+1)*(2*l2+1)*N1*N2);

      if(dev==0){

	auto& C=SO3_cgbank.getf(CGindex(l1,l2,l));
	batched_cmvprod(_r,_x,_y,[&](const cnine::Ctensor2_view& _r, const cnine::Ctensor2_view& _x, const cnine::Ctensor2_view& _y){
	    SO3part2_view r(_r);
	    SO3part2_view x(_x);
	    SO3part2_view y(_y);

	    int offs=_offs;
	    for(int n1=0; n1<N1; n1++){
	      for(int n2=0; n2<N2; n2++){
		for(int m1=-l1; m1<=l1; m1++){
		  for(int m2=std::max(-l2,-l-m1); m2<=std::min(l2,l-m1); m2++){
		    r.inc(m1+m2,offs+n2,C(m1+l1,m2+l2)*x(m1,n1)*y(m2,n2));
		  }
		}
	      }
	      offs+=N2;
	    }
	  });

      }
      //else CUDA_STREAM(SO3partB_addReducingCGproduct_cu(_r,_x,_y,_offs,stream));

    }

  };



}

#endif

      //LoggedTimer timer("  CGproduct("+to_string(l1)+","+to_string(l2)+","+to_string(l)+")[b="+
      //to_string(B)+",n1="+to_string(N1)+",n2="+to_string(N2)+",dev="+to_string(dev)+"]",B*(2*l1+1)*(2*l2+1)*N1*N2);
