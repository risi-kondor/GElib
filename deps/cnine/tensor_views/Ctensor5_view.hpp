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


#ifndef _CnineCtensor5_view
#define _CnineCtensor5_view

#include "Cnine_base.hpp"
#include "Gstrides.hpp"
#include "MultiLoop.hpp"
#include "Rtensor6_view.hpp"
#include "Ctensor4_view.hpp"

namespace cnine{


  class Ctensor5_view{
  public:

    float* arr;
    float* arrc;
    int n0,n1,n2,n3,n4;
    int s0,s1,s2,s3,s4;
    int dev=0;

  public:

    Ctensor5_view(){}

    Ctensor5_view(float* _arr, float* _arrc): 
      arr(_arr), arrc(_arrc){}

    Ctensor5_view(float* _arr, float* _arrc, 
      const int _n0, const int _n1, const int _n2, const int _n3, const int _n4,  
      const int _s0, const int _s1, const int _s2, const int _s3, const int _s4, const int _dev=0): 
      arr(_arr), arrc(_arrc), n0(_n0), n1(_n1), n2(_n2), n3(_n3), n4(_n4), 
      s0(_s0), s1(_s1), s2(_s2), s3(_s3), s4(_s4), dev(_dev){}

    Ctensor5_view(float* _arr, const int _n0, const int _n1, const int _n2, const int _n3, const int _n4, 
      const int _s0, const int _s1, const int _s2, const int _s3, const int _s4, const int _coffs=1): 
      arr(_arr), arrc(_arr+_coffs), n0(_n0), n1(_n1), n2(_n2), n3(_n3), n4(_n4), s0(_s0), s1(_s1), s2(_s2), s3(_s3), s4(_s4){}

    Ctensor5_view(float* _arr,  const Gdims& _dims, const Gstrides& _strides, const int _coffs=1, const int _dev=0):
      arr(_arr), arrc(_arr+_coffs), dev(_dev){
      assert(_dims.size()==5);
      n0=_dims[0];
      n1=_dims[1];
      n2=_dims[2];
      n3=_dims[3];
      n4=_dims[4];
      s0=_strides[0];
      s1=_strides[1];
      s2=_strides[2];
      s3=_strides[3];
      s4=_strides[4];
    }


  public: // ---- Conversions -------------------------------------------------------------------------------


    Rtensor6_view as_real() const{
      CNINE_ASSRT(arrc==arr+1);
      return Rtensor6_view(arr,n0,n1,n2,n3,n4,2,s0,s1,s2,s3,s4,1,dev);
    }


  public: // ---- Access ------------------------------------------------------------------------------------


    Gdims get_dims() const{
      return Gdims({n0,n1,n2,n3,n4});
    }

    complex<float> operator()(const int i0, const int i1, const int i2, const int i3, const int i4) const{
      int t=s0*i0+s1*i1+s2*i2+s3*i3+s4*i4;
      return complex<float>(arr[t],arrc[t]);
    }

    void set(const int i0, const int i1, const int i2, const int i3, const int i4, complex<float> x) const{
      int t=s0*i0+s1*i1+s2*i2+s3*i3+s4*i4;
      arr[t]=std::real(x);
      arrc[t]=std::imag(x);
    }

    void inc(const int i0, const int i1, const int i2, const int i3, const int i4, complex<float> x) const{
      int t=s0*i0+s1*i1+s2*i2+s3*i3+s4*i4;
      arr[t]+=std::real(x);
      arrc[t]+=std::imag(x);
    }


  public: // ---- Lambdas ------------------------------------------------------------------------------------


    template<typename VIEW>
    void foreach_slice0(std::function<void(const Ctensor4_view& self_slice, const Ctensor2_view& x_slice)> lambda, 
      const VIEW& x) const{
      //const _bind0<VIEW>& _x){
      //const VIEW& x=_x.obj;
      assert(n0==x.n0);
      for(int i=0; i<n0; i++)
	lambda(slice0(i),x.slice0(i));
    }



  public: // ---- Cumulative operations ---------------------------------------------------------------------

    /*
   // Product type: abcd,be -> aecd
    void add_mix_1_0(const Ctensor5_view& x, const Ctensor2_view& y){
      fuse23().add_mix_1_0(x.fuse23(),y);
    }

    void add_mix_1_H(const Ctensor5_view& x, const Ctensor2_view& y){
      fuse23().add_mix_1_H(x.fuse23(),y);
    }


   // Product type: abcd,de -> abce
    void add_mix_3_0(const Ctensor5_view& x, const Ctensor2_view& y) const{
      fuse01().add_mix_2_0(x.fuse01(),y);
    }

    void add_mix_3_H(const Ctensor5_view& x, const Ctensor2_view& y) const{
      fuse01().add_mix_2_H(x.fuse01(),y);
    }


   // Product type: abc,bdc -> abdc
    void add_expand_2(const Ctensor3_view& x, const Ctensor3_view& y){
      CNINE_CHECK_DEV3((*this),x,y);
      assert(x.n0==n0);
      assert(x.n1==n1);
      assert(x.n2==n3);
      assert(y.n0==x.n1);
      assert(y.n1==n2);
      assert(y.n2==x.n2);

      BasicCproduct_4<float>(x.arr,x.arrc,y.arr,y.arrc,arr,arrc,
	n0,n1,n2,n3, x.s0,x.s1,0,x.s2, 0,y.s0,y.s1,y.s2, s0,s1,s2,s3, dev);

    }


   // Product type: abic,bic -> abc
    void add_contract_abic_bic_abc_to(const Ctensor3_view& r, const Ctensor3_view& y) const{
      foreach_slice0([&](const Ctensor3_view& xslice, const Ctensor2_view& rslice){
	  xslice.add_contract_aib_aib_ab_to(rslice,y);
	},r);
      return; 
    }
    */

  public: // ---- Other views -------------------------------------------------------------------------------


    Ctensor4_view slice0(const int i) const{
      return Ctensor4_view(arr+i*s0,arrc+i*s0,n1,n2,n3,n4,s1,s2,s3,s4,dev);
    }

    Ctensor4_view slice1(const int i){
      return Ctensor4_view(arr+i*s1,arrc+i*s1,n0,n2,n3,n4,s0,s2,s3,s4,dev);
    }

    Ctensor4_view slice2(const int i){
      return Ctensor4_view(arr+i*s2,arrc+i*s2,n0,n1,n3,n4,s0,s1,s3,s4,dev);
    }

    Ctensor4_view slice3(const int i){
      return Ctensor4_view(arr+i*s3,arrc+i*s3,n0,n1,n2,n4,s0,s1,s2,s4,dev);
    }

    Ctensor4_view slice4(const int i){
      return Ctensor4_view(arr+i*s4,arrc+i*s4,n0,n1,n2,n3,s0,s1,s2,s3,dev);
    }


    Ctensor4_view fuse01() const{
      return Ctensor4_view(arr,arrc,n0*n1,n2,n3,n4,s1,s2,s3,s4,dev);
    }    

    Ctensor4_view fuse12() const{
      return Ctensor4_view(arr,arrc,n0,n1*n2,n3,n4,s0,s2,s3,s4,dev);
    }    

    Ctensor4_view fuse23() const{
      return Ctensor4_view(arr,arrc,n0,n1,n2*n3,n4,s0,s1,s3,s4,dev);
    }    

    Ctensor4_view fuse34() const{
      return Ctensor4_view(arr,arrc,n0,n1,n2,n3*n4,s0,s1,s2,s4,dev);
    }    

  };


  // The last two dimensions are tensor dims
  inline void batched_cmvprod(const Ctensor4_view& r, const Ctensor5_view& x, const Ctensor4_view& y, 
    const std::function<void(const Ctensor2_view&, const Ctensor2_view&, const Ctensor2_view&)>& lambda){
      CNINE_CHECK_BATCH3(r,x,y);
      CNINE_ASSRT(x.n1==r.n1);
      CNINE_ASSRT(x.n2==y.n1);
      const int J=x.n1;
      const int K=x.n2;

      MultiLoop(r.n0,[&](const int b){
	  Ctensor3_view rb=r.slice0(b);
	  Ctensor4_view xb=x.slice0(b);
	  Ctensor3_view yb=y.slice0(b);

	  for(int j=0; j<J; j++){
	    Ctensor2_view R=rb.slice0(j);
	    Ctensor3_view X=xb.slice0(j);

	    for(int k=0; k<K; k++)
	      lambda(R,X.slice0(k),yb.slice0(k));
	  }

	});
    }




}


#endif 
 
