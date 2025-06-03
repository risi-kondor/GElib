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


#ifndef _CnineCtensor3_view
#define _CnineCtensor3_view

#include "Cnine_base.hpp"
#include "Gstrides.hpp"
#include "GstridesB.hpp"

#include "Ctensor2_view.hpp"
#include "BasicCtensorProducts.hpp"
#include "Rtensor4_view.hpp"


namespace cnine{

  #ifdef _WITH_CUDA
  extern void Rtensor_add_cu(const Rtensor3_view& r, const Rtensor3_view& x, const cudaStream_t& stream);
  //extern void Rtensor_add_cu(const Rtensor3_view& r, const Rtensor3_view& x, const float c, const cudaStream_t& stream);
  #endif 

  class Ctensor3_view{
  public:

    float* arr;
    float* arrc;
    int n0,n1,n2;
    int s0,s1,s2;
    int dev=0;

  public:

    Ctensor3_view(){}

    Ctensor3_view(float* _arr, float* _arrc): 
      arr(_arr), arrc(_arrc){}

    Ctensor3_view(float* _arr, float* _arrc, 
      const int _n0, const int _n1, const int _n2, const int _s0, const int _s1, const int _s2, const int _dev=0): 
      arr(_arr), arrc(_arrc), n0(_n0), n1(_n1), n2(_n2), s0(_s0), s1(_s1), s2(_s2), dev(_dev){}

    Ctensor3_view(float* _arr, const int _n0, const int _n1, const int _n2, 
      const int _s0, const int _s1, const int _s2, const int _coffs=1, const int _dev=0): 
      arr(_arr), arrc(_arr+_coffs), n0(_n0), n1(_n1), n2(_n2), s0(_s0), s1(_s1), s2(_s2), dev(_dev){}

    Ctensor3_view(float* _arr,  const Gdims& _dims, const Gstrides& _strides, const int _coffs=1, const int _dev=0):
      arr(_arr), arrc(_arr+_coffs), dev(_dev){
      assert(_dims.size()==3);
      n0=_dims[0];
      n1=_dims[1];
      n2=_dims[2];
      s0=_strides[0];
      s1=_strides[1];
      s2=_strides[2];
    }

    Ctensor3_view(float* _arr,  const Gdims& _dims, const GstridesB& _strides, const int _coffs=1, const int _dev=0):
      arr(_arr), arrc(_arr+_coffs), dev(_dev){
      assert(_dims.size()==3);
      n0=_dims[0];
      n1=_dims[1];
      n2=_dims[2];
      s0=_strides[0];
      s1=_strides[1];
      s2=_strides[2];
    }

    Ctensor3_view(float* _arr, const Gdims& _dims, const Gstrides& _strides, 
      const GindexSet& a, const GindexSet& b, const GindexSet& c, const int _coffs=1, const int _dev=0):
      arr(_arr), arrc(_arr+_coffs), dev(_dev){
      assert(_strides.is_regular(_dims));
      assert(a.is_contiguous());
      assert(b.is_contiguous());
      assert(c.is_contiguous());
      assert(a.is_disjoint(b));
      assert(a.is_disjoint(c));
      assert(b.is_disjoint(c));
      assert(a.covers(_dims.size(),b,c));
      n0=_dims.unite(a);
      n1=_dims.unite(b);
      n2=_dims.unite(c);
      s0=_strides[a.back()];
      s1=_strides[b.back()];
      s2=_strides[c.back()];
    }

  public: // ---- Conversions -------------------------------------------------------------------------------


    Rtensor4_view as_real() const{
      CNINE_ASSRT(arrc==arr+1);
      return Rtensor4_view(arr,n0,n1,n2,2,s0,s1,s2,1,dev);
    }


  public: // ---- Access ------------------------------------------------------------------------------------


    complex<float> operator()(const int i0, const int i1, const int i2) const{
      int t=s0*i0+s1*i1+s2*i2;
      return complex<float>(arr[t],arrc[t]);
    }

    void set(const int i0, const int i1, const int i2, complex<float> x) const{
      int t=s0*i0+s1*i1+s2*i2;
      arr[t]=std::real(x);
      arrc[t]=std::imag(x);
    }

    void inc(const int i0, const int i1, const int i2, complex<float> x) const{
      int t=s0*i0+s1*i1+s2*i2;
      arr[t]+=std::real(x);
      arrc[t]+=std::imag(x);
    }

    bool is_regular() const{
      if(arrc-arr!=1) return false;
      if(s2!=2) return false;
      if(s1!=s2*n2) return false;
      if(s0!=s1*n1) return false;
      return true;
    }


  public: // ---- foreach ------------------------------------------------------------------------------------


    template<typename VIEW, typename SLICE>
    void foreach_slice0(std::function<void(const Ctensor2_view& self_slice, const SLICE& x_slice)> lambda, 
      const _bind0<VIEW>& _x){
      assert(n0==_x.obj.n0);
      for(int i=0; i<n0; i++)
	lambda(slice0(i),_x.obj.slice0(i));
    }


  public: // ---- Cumulative operations ---------------------------------------------------------------------


    void add(const Ctensor3_view& x) const{ // TODO 
      CNINE_UNIMPL();
      CNINE_DEVICE_SAME(x);
      CNINE_ASSRT(x.n0==n0);
      CNINE_ASSRT(x.n1==n1);
      if(x.n0*x.n1==0) return;
      if(is_regular() && x.is_regular()){
	CPUCODE(stdadd<float>(x.arr,x.arr+n0*s0,arr));
	GPUCODE(const float alpha=1; CUBLAS_SAFE(cublasSaxpy(cnine_cublas,n0*n1,&alpha,x.arr,1,arr,1)));
      }else{
	CPUCODE(for(int i0=0; i0<x.n0; i0++) for(int i1=0; i1<x.n1; i1++) for(int i2=0; i2<x.n2; i2++) {inc(i0,i1,i2,x(i0,i1,i2));});
	CNINE_CPUONLY();
	//GPUCODE(CUDA_STREAM(Rtensor_add_cu(*this,x,stream)));
      }
    }


    // Product type: aic,ib -> abc
    void add_mix_1_0(const Ctensor3_view& x, const Ctensor2_view& y){
      CNINE_CHECK_DEV3((*this),x,y);

      const int I=x.n1;
      assert(y.n0==I);
      assert(x.n0==n0);
      assert(y.n1==n1);
      assert(x.n2==n2);

      if(dev==0){
	for(int a=0; a<n0; a++)
	  for(int b=0; b<n1; b++)
	    for(int c=0; c<n2; c++){
	      complex<float> t=0;
	      for(int i=0; i<I; i++)
		t+=x(a,i,c)*y(i,b);
	      inc(a,b,c,t);
	    }
      }

      if(dev==1){
	assert(is_regular()); // stride this!!
	#ifdef _WITH_CUBLAS
	//cout<<x.arr<<endl;
	//cout<<y.arr<<endl;
	//cout<<arr<<endl;
	//cout<<n2<<n1<<y.n1<<x.n2<<n2<<endl;

	cuComplex alpha;
	alpha.x=1.0f;
	alpha.y=0.0f;
	CUBLAS_SAFE(cublasCgemmStridedBatched(cnine_cublas,CUBLAS_OP_N,CUBLAS_OP_T,n2,n1,x.n1,&alpha,
	  reinterpret_cast<cuComplex*>(x.arr),x.n2,x.s0/2,
	  reinterpret_cast<cuComplex*>(y.arr),y.n1,0,&alpha,
	  reinterpret_cast<cuComplex*>(arr),n2,s0/2,n0)); 
	//CUBLAS_SAFE(cublasCgemm(cnine_cublas,CUBLAS_OP_N,CUBLAS_OP_T,n2,n1,x.n1,&alpha,
	//  reinterpret_cast<cuComplex*>(x.arr),x.n2,
	//  reinterpret_cast<cuComplex*>(y.arr),y.n1,&alpha,
	//  reinterpret_cast<cuComplex*>(arr),n2)); 
	#endif
      }
    }

    
    // Product type: aic,(ib)^H -> abc
    void add_mix_1_H(const Ctensor3_view& x, const Ctensor2_view& y){
      CNINE_CHECK_DEV3((*this),x,y);

      const int I=x.n1;
      assert(y.n1==I);
      assert(x.n0==n0);
      assert(y.n0==n1);
      assert(x.n2==n2);

      if(dev==0){
	for(int a=0; a<n0; a++)
	  for(int b=0; b<n1; b++)
	    for(int c=0; c<n2; c++){
	      complex<float> t=0;
	      for(int i=0; i<I; i++)
		t+=x(a,i,c)*std::conj(y(b,i));
	      inc(a,b,c,t);
	    }
      }

      if(dev==1){
	CNINE_UNIMPL();
	assert(is_regular()); // stride this!!
	CNINE_CPUONLY();
	#ifdef _WITH_CUBLAS
	//cuComplex alpha;
	//alpha.x=1.0f;
	//alpha.y=0.0f;
	//CUBLAS_SAFE(cublasCgemmStridedBatched(cnine_cublas,CUBLAS_OP_H,CUBLAS_OP_N,n1,n0,x.n1,&alpha,
	//  reinterpret_cast<cuComplex*>(y.arr),y.n1,0,
	//  reinterpret_cast<cuComplex*>(x.arr),x.n1,x.s2,&alpha,
	//    reinterpret_cast<cuComplex*>(arr),n1,s2)); 
	#endif
      }
    }
    

    // Product type: abi,ic -> abc
    void add_mix_2_0(const Ctensor3_view& x, const Ctensor2_view& y){
      fuse01().add_matmul(x.fuse01(),y);
    }
    
    void add_mix_2_H(const Ctensor3_view& x, const Ctensor2_view& y){
      fuse01().add_matmul_AH(x.fuse01(),y);
    }
    

   // Product type: aib,aib -> ab
    void add_contract_aib_aib_ab_to(const Ctensor2_view& r, const Ctensor3_view& y) const{
      CNINE_CHECK_DEV3((*this),r,y); 
      assert(r.n0==n0);
      assert(r.n1==n2);
      assert(y.n0==n0);
      assert(y.n1==n1);
      assert(y.n2==n2);

      BasicCproduct_2_1<float>(arr,arrc,y.arr,y.arrc,r.arr,r.arrc,n0,n2,n1,s0,s2,s1,y.s0,y.s2,y.s1,r.s0,r.s1,dev);
      return;

    }


  public: // ---- Other views -------------------------------------------------------------------------------


    Ctensor2_view slice0(const int i) const{
      return Ctensor2_view(arr+i*s0,arrc+i*s0,n1,n2,s1,s2,dev);
    }

    Ctensor2_view slice1(const int i) const{
      return Ctensor2_view(arr+i*s1,arrc+i*s1,n0,n2,s0,s2,dev);
    }

    Ctensor2_view slice2(const int i) const{
      return Ctensor2_view(arr+i*s2,arrc+i*s2,n0,n1,s0,s1,dev);
    }

    Ctensor2_view fuse01() const{
      return Ctensor2_view(arr,arrc,n0*n1,n2,s1,s2,dev);
    }    

    Ctensor2_view fuse12() const{
      return Ctensor2_view(arr,arrc,n0,n1*n2,s0,s2,dev);
    }    


  public: // ---- Conversions -------------------------------------------------------------------------------

    
    Gtensor<complex<float> > gtensor() const{
      Gtensor<complex<float> > R({n0,n1,n2},fill::raw);
      //cout<<"R"<<R.dims<<endl;
      for(int i0=0; i0<n0; i0++)
	for(int i1=0; i1<n1; i1++)
	  for(int i2=0; i2<n2; i2++){
	    R(i0,i1,i2)=(*this)(i0,i1,i2);
	  }
      return R;
    }
    

  public: // ---- Operations ---------------------------------------------------------------------------------


    Ctensor3_view flip() const{
      Ctensor3_view R(*this);
      R.arr=arr+(n1-1)*s1+(n2-1)*s2;
      R.arrc=arrc+(n1-1)*s1+(n2-1)*s2;
      R.s1=-s1;
      R.s2=-s2;
      return R;
    }

    
  public: // ---- I/O ----------------------------------------------------------------------------------------

  
    string str(const string indent="") const{
      return gtensor().str(indent);
    }

    friend ostream& operator<<(ostream& stream, const Ctensor3_view& x){
      stream<<x.str(); return stream;
    }

    

  };


  inline Ctensor3_view unsqueeze0(const Ctensor2_view& x){
    return Ctensor3_view(x.arr,x.arrc,1,x.n0,x.n1,x.s0*x.n0,x.s0,x.s1,x.dev);
  }

  inline Ctensor3_view unsqueeze1(const Ctensor2_view& x){
    return Ctensor3_view(x.arr,x.arrc,x.n0,1,x.n1,x.s1,x.s1,x.s1,x.dev);
  }

  inline Ctensor3_view unsqueeze2(const Ctensor2_view& x){
    return Ctensor3_view(x.arr,x.arrc,x.n0,x.n1,1,x.s0,x.s1,x.s1,x.dev);
  }



  /*
  class Ctensor3_view_t2: public Ctensor3_view{
  public:

    int tsize;
    int nt;
    int st;

    Ctensor3_view_t2(const Ctensor3_view& x, const int n):
      Ctensor3_view(x), tsize(n), nt((x.n2-1)/n+1), st(x.s2*n){
    }

  };
  */

}


#endif 

      /*
      assert(x.dev==0);
      assert(y.dev==0);
      assert(dev==0);

      const int I=x.n2;
      assert(y.n0==I);
      assert(x.n0==n0);
      assert(x.n1==n1);
      assert(y.n1==n2);

      for(int a=0; a<n0; a++)
	for(int b=0; b<n1; b++)
	  for(int c=0; c<n2; c++){
	    complex<float> t=0;
	    for(int i=0; i<I; i++)
	      t+=x(a,b,i)*y(i,c);
	    inc(a,b,c,t);
	  }
      */
    /*
    void add_mprod2(const Ctensor2_view& x, const Ctensor2_view& M) const{
      const int B=n0;
      asset(x.n0==B);
      cnine::MultiLoop(B,[&](const int b){
	  slice0(b).add_mprod2(x.slice0(b),M);});
    }
    */

      /*
      if(dev==0){// Deprecated 
	for(int a=0; a<r.n0; a++)
	  for(int b=0; b<r.n1; b++){
	      complex<float> t=0;
	      for(int i=0; i<n1; i++)
		t+=(*this)(a,i,b)*y(a,i,b);
	      r.inc(a,b,t);
	    }
      }

      if(dev==1){
	CNINE_CPUONLY();
      }
      */

