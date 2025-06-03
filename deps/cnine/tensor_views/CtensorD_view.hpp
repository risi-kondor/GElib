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


#ifndef _CnineCtensorD_view
#define _CnineCtensorD_view

#include "Cnine_base.hpp"
#include "Gstrides.hpp"
#include "Gtensor.hpp"

#include "Ctensor3_view.hpp"


namespace cnine{


  class CtensorD_view{
  public:

    float* arr;
    float* arrc;
    const Gdims dims;
    const Gstrides strides;
    int dev=0;

  public:

    CtensorD_view(){}

    CtensorD_view(float* _arr, float* _arrc): 
      arr(_arr), arrc(_arrc){}

    CtensorD_view(float* _arr, float* _arrc, const int _n0, const int _s0, const int _dev=0): 
      arr(_arr), arrc(_arrc), dims(_n0), strides(_s0), dev(_dev){}

    CtensorD_view(float* _arr, float* _arrc, const int _n0, const int _n1, const int _s0, const int _s1, const int _dev=0): 
      arr(_arr), arrc(_arrc), dims({_n0,_n1}), strides({_s0,_s1}), dev(_dev){}

    CtensorD_view(float* _arr, float* _arrc, const int _n0, const int _n1, const int _n2, 
      const int _s0, const int _s1, const int _s2, const int _dev=0): 
      arr(_arr), arrc(_arrc), dims({_n0,_n1,_n2}), strides({_s0,_s1,_s2}), dev(_dev){}

    CtensorD_view(float* _arr, float* _arrc, const int _n0, const int _n1, const int _n2, const int _n3,
      const int _s0, const int _s1, const int _s2, const int _s3, const int _dev=0): 
      arr(_arr), arrc(_arrc), dims({_n0,_n1,_n2,_n3}), strides({_s0,_s1,_s2,_s3}), dev(_dev){}

    CtensorD_view(float* _arr, float* _arrc,  const Gdims& _dims, const Gstrides& _strides, const int _dev=0):
      arr(_arr), arrc(_arrc), dims(_dims), strides(_strides), dev(_dev){
    }



  public: // ---- Getters ------------------------------------------------------------------------------------


    complex<float> operator()(const int i0) const{
      CNINE_CHECK_RANGE(if(i0<0 || i0>=dims[0]) 
	  throw std::out_of_range("cnine::CtensorD_view: index "+Gindex({i0}).str()+" out of range of view size "+dims.str()));
      return complex<float>(arr[strides.offs(i0)],arrc[strides.offs(i0)]);
    }

    complex<float> operator()(const int i0, const int i1) const{
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i0>=dims[0] || i1>=dims[1]) 
	  throw std::out_of_range("cnine::CtensorD_view: index "+Gindex({i0,i1}).str()+" out of range of view size "+dims.str()));
      return complex<float>(arr[strides.offs(i0,i1)],arrc[strides.offs(i0,i1)]);
    }

    complex<float> operator()(const int i0, const int i1, const int i2) const{
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i2<0 || i0>=dims[0] || i1>=dims[1] || i2>=dims[2]) 
	  throw std::out_of_range("cnine::CtensorD_view: index "+Gindex({i0,i1,i2}).str()+" out of range of view size "+dims.str()));
      return complex<float>(arr[strides.offs(i0,i1,i2)],arrc[strides.offs(i0,i1,i2)]);
    }

    complex<float> operator()(const int i0, const int i1, const int i2, const int i3) const{
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i2<0 || i3<0 || i0>=dims[0] || i1>=dims[1] || i2>=dims[2] || i3>=dims[3]) 
	  throw std::out_of_range("cnine::CtensorD_view: index "+Gindex({i0,i1,i2,i3}).str()+" out of range of view size "+dims.str()));
      return complex<float>(arr[strides.offs(i0,i1,i2,i3)],arrc[strides.offs(i0,i1,i2,i3)]);
    }

    complex<float> operator()(const Gindex& ix) const{
      CNINE_CHECK_RANGE(ix.check_range(dims));
      int t=ix(strides);
      return complex<float>(arr[t],arrc[t]);
    }


  public: // ---- Setters ------------------------------------------------------------------------------------


    void set(const int i0, complex<float> x){
      CNINE_CHECK_RANGE(if(i0<0 || i0>=dims[0]) 
	  throw std::out_of_range("cnine::CtensorD_view: index "+Gindex({i0}).str()+" out of range of view size "+dims.str()));
      arr[strides.offs(i0)]=std::real(x);
      arrc[strides.offs(i0)]=std::imag(x);
    }

    void set(const int i0, const int i1, complex<float> x){
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i0>=dims[0] || i1>=dims[1]) 
	  throw std::out_of_range("cnine::CtensorD_view: index "+Gindex({i0,i1}).str()+" out of range of view size "+dims.str()));
      arr[strides.offs(i0,i1)]=std::real(x);
      arrc[strides.offs(i0,i1)]=std::imag(x);
    }

    void set(const int i0, const int i1, const int i2, complex<float> x){
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i2<0 || i0>=dims[0] || i1>=dims[1] || i2>=dims[2]) 
	  throw std::out_of_range("cnine::CtensorD_view: index "+Gindex({i0,i1,i2}).str()+" out of range of view size "+dims.str()));
      arr[strides.offs(i0,i1,i2)]=std::real(x);
      arrc[strides.offs(i0,i1,i2)]=std::imag(x);
    }

    void set(const int i0, const int i1, const int i2, const int i3, complex<float> x){
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i2<0 || i3<0 || i0>=dims[0] || i1>=dims[1] || i2>=dims[2] || i3>=dims[3]) 
	  throw std::out_of_range("cnine::CtensorD_view: index "+Gindex({i0,i1,i2,i3}).str()+" out of range of view size "+dims.str()));
      arr[strides.offs(i0,i1,i2,i3)]=std::real(x);
      arrc[strides.offs(i0,i1,i2,i3)]=std::imag(x);
    }


  public: // ---- Incrementers -------------------------------------------------------------------------------


    void inc(const int i0, complex<float> x){
      CNINE_CHECK_RANGE(if(i0<0 || i0>=dims[0]) 
	  throw std::out_of_range("cnine::CtensorD_view: index "+Gindex({i0}).str()+" out of range of view size "+dims.str()));
      arr[strides.offs(i0)]+=std::real(x);
      arrc[strides.offs(i0)]+=std::imag(x);
    }

    void inc(const int i0, const int i1, complex<float> x){
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i0>=dims[0] || i1>=dims[1]) 
	  throw std::out_of_range("cnine::CtensorD_view: index "+Gindex({i0,i1}).str()+" out of range of view size "+dims.str()));
      arr[strides.offs(i0,i1)]+=std::real(x);
      arrc[strides.offs(i0,i1)]+=std::imag(x);
    }

    void inc(const int i0, const int i1, const int i2, complex<float> x){
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i2<0 || i0>=dims[0] || i1>=dims[1] || i2>=dims[2]) 
	  throw std::out_of_range("cnine::CtensorD_view: index "+Gindex({i0,i1,i2}).str()+" out of range of view size "+dims.str()));
      arr[strides.offs(i0,i1,i2)]+=std::real(x);
      arrc[strides.offs(i0,i1,i2)]+=std::imag(x);
    }

    void inc(const int i0, const int i1, const int i2, const int i3, complex<float> x){
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i2<0 || i3<0 || i0>=dims[0] || i1>=dims[1] || i2>=dims[2] || i3>=dims[3]) 
	  throw std::out_of_range("cnine::CtensorD_view: index "+Gindex({i0,i1,i2,i3}).str()+" out of range of view size "+dims.str()));
      arr[strides.offs(i0,i1,i2,i3)]+=std::real(x);
      arrc[strides.offs(i0,i1,i2,i3)]+=std::imag(x);
    }




    //CtensorD_view block(const int i0, const int i1, const int i2, const int i3, const int m0, const int m1, const int m2, const int m3) const{
    //return CtensorD_view(arr+i0*s0+i1*s1+i2*s2+i3*s3,m0,m1,m2,m3,s0,s1,s2,s3,dev);
    //}


  public: // ---- Cumulative operations ---------------------------------------------------------------------

    /*
    void add(const CtensorD_view& y){
      assert(y.n0==n0);
      assert(y.n1==n1);
      assert(y.n2==n2);
      assert(y.n3==n3);
      for(int i0=0; i0<n0; i0++)
	for(int i1=0; i1<n1; i1++)
	  for(int i2=0; i2<n2; i2++)
	    for(int i3=0; i3<n3; i3++)
	      inc(i0,i1,i2,i3,y(i0,i1,i2,i3));
    }
    */


  public: // ---- Other views -------------------------------------------------------------------------------


    Ctensor2_view fuse0X() const{
      if(dims.size()==0) return Ctensor2_view(arr,arrc,1,1,0,0,dev);
      if(dims.size()==1) return Ctensor2_view(arr,arrc,dims[0],1,strides[0],0,dev);
      return Ctensor2_view(arr,arrc,dims[0],strides[0]/strides.back(),strides[0],strides.back(),dev);
    }    


    CtensorD_view slice0(const int i) const{
      CNINE_CHECK_RANGE(if(i<0 || i>=dims[0]) throw std::out_of_range("cnine::CtensorD_view:slice0(int): index "+to_string(i)+" out of range of [0,"+to_string(dims[0]-1)+"]"));
      return CtensorD_view(arr+i*strides[0],arrc+i*strides[0],dims.chunk(1),strides.chunk(1),dev);
    }

    /*
    Ctensor3_view slice1(const int i){
      CNINE_CHECK_RANGE(if(i<0 || i>=n1) 
	  throw std::out_of_range("cnine::CtensorD_view:slice1(int): index "+to_string(i)+" out of range of [0,"+to_string(n1-1)+"]");)
	return Ctensor3_view(arr+i*s1,n0,n2,n3,s0,s2,s3,dev);
    }

    Ctensor3_view slice2(const int i){
      CNINE_CHECK_RANGE(if(i<0 || i>=n2) 
	  throw std::out_of_range("cnine::CtensorD_view:slice2(int): index "+to_string(i)+" out of range of [0,"+to_string(n2-1)+"]");)
	return Ctensor3_view(arr+i*s2,n0,n1,n3,s0,s1,s3,dev);
    }

    Ctensor3_view slice3(const int i){
      CNINE_CHECK_RANGE(if(i<0 || i>=n3) 
	  throw std::out_of_range("cnine::CtensorD_view:slice3(int): index "+to_string(i)+" out of range of [0,"+to_string(n3-1)+"]");)
	return Ctensor3_view(arr+i*s3,n0,n1,n2,s0,s1,s2,dev);
    }

    Ctensor3_view fuse01(){
      return Ctensor3_view(arr,n0*n1,n2,n3,s1,s2,s3,dev);
    }    

    Ctensor3_view fuse12(){
      return Ctensor3_view(arr,n0,n1*n2,n3,s0,s2,s3,dev);
    }    

    Ctensor3_view fuse23(){
      return Ctensor3_view(arr,n0,n1,n2*n3,s0,s1,s3,dev);
    }    


    Ctensor2_view slice01(const int i, const int j){
      CNINE_CHECK_RANGE(if(i<0 || i>=n0) 
	  throw std::out_of_range("cnine::CtensorD_view:slice01(int): index "+to_string(i)+" out of range of [0,"+to_string(n0-1)+"]");)
      CNINE_CHECK_RANGE(if(i<0 || i>=n1) 
	  throw std::out_of_range("cnine::CtensorD_view:slice01(int): index "+to_string(i)+" out of range of [0,"+to_string(n1-1)+"]");)
	return Ctensor2_view(arr+i*s0+j*s1,n2,n3,s2,s3,dev);
    }

    Ctensor2_view slice02(const int i, const int j){
      CNINE_CHECK_RANGE(if(i<0 || i>=n0) 
	  throw std::out_of_range("cnine::CtensorD_view:slice02(int): index "+to_string(i)+" out of range of [0,"+to_string(n0-1)+"]");)
      CNINE_CHECK_RANGE(if(i<0 || i>=n2) 
	  throw std::out_of_range("cnine::CtensorD_view:slice02(int): index "+to_string(i)+" out of range of [0,"+to_string(n2-1)+"]");)
	return Ctensor2_view(arr+i*s0+j*s2,n1,n3,s1,s3,dev);
    }

    Ctensor2_view slice03(const int i, const int j){
      CNINE_CHECK_RANGE(if(i<0 || i>=n0) 
	  throw std::out_of_range("cnine::CtensorD_view:slice03(int): index "+to_string(i)+" out of range of [0,"+to_string(n0-1)+"]");)
      CNINE_CHECK_RANGE(if(i<0 || i>=n3) 
	  throw std::out_of_range("cnine::CtensorD_view:slice03(int): index "+to_string(i)+" out of range of [0,"+to_string(n3-1)+"]");)
	return Ctensor2_view(arr+i*s0+j*s3,n1,n2,s1,s2,dev);
    }

    Ctensor2_view slice12(const int i, const int j){
      CNINE_CHECK_RANGE(if(i<0 || i>=n1) 
	  throw std::out_of_range("cnine::CtensorD_view:slice12(int): index "+to_string(i)+" out of range of [0,"+to_string(n1-1)+"]");)
	CNINE_CHECK_RANGE(if(i<0 || i>=n2) 
	    throw std::out_of_range("cnine::CtensorD_view:slice12(int): index "+to_string(i)+" out of range of [0,"+to_string(n2-1)+"]");)
	return Ctensor2_view(arr+i*s1+j*s2,n0,n3,s0,s3,dev);
    }

    Ctensor2_view slice13(const int i, const int j){
      CNINE_CHECK_RANGE(if(i<0 || i>=n1) 
	  throw std::out_of_range("cnine::CtensorD_view:slice13(int): index "+to_string(i)+" out of range of [0,"+to_string(n1-1)+"]");)
      CNINE_CHECK_RANGE(if(i<0 || i>=n3) 
	  throw std::out_of_range("cnine::CtensorD_view:slice13(int): index "+to_string(i)+" out of range of [0,"+to_string(n3-1)+"]");)
	return Ctensor2_view(arr+i*s1+j*s3,n0,n2,s0,s2,dev);
    }

    Ctensor2_view slice23(const int i, const int j){
      CNINE_CHECK_RANGE(if(i<0 || i>=n2) 
	  throw std::out_of_range("cnine::CtensorD_view:slice23(int): index "+to_string(i)+" out of range of [0,"+to_string(n2-1)+"]");)
      CNINE_CHECK_RANGE(if(i<0 || i>=n3) 
	  throw std::out_of_range("cnine::CtensorD_view:slice23(int): index "+to_string(i)+" out of range of [0,"+to_string(n3-1)+"]");)
	return Ctensor2_view(arr+i*s2+j*s3,n0,n1,s0,s1,dev);
    }
    */


  public: // ---- Conversions -------------------------------------------------------------------------------


    Gtensor<complex<float> > gtensor() const{
      assert(dev==0);
      Gtensor<complex<float> > R(dims,fill::raw);
      for(int i=0; i<R.asize; i++)
	R.arr[i]=(*this)(Gindex(2*i,strides));
      return R;
    }
   
    
    
  public: // ---- I/O ----------------------------------------------------------------------------------------

  
    string str(const string indent="") const{
      return gtensor().str(indent);
    }

    friend ostream& operator<<(ostream& stream, const CtensorD_view& x){
      stream<<x.str(); return stream;
    }


  };

}


#endif 
