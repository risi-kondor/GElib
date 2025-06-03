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


#ifndef _CnineCtensorView
#define _CnineCtensorView

#include "Cnine_base.hpp"
#include "Gstrides.hpp"
#include "Gtensor.hpp"
#include "RtensorView.hpp"

#include "Ctensor2_view.hpp"


namespace cnine{


  class CtensorView{
  public:

    float* arr;
    float* arrc;
    const Gdims dims;
    const Gstrides strides;
    int dev=0;

  public:

    CtensorView(){}

    CtensorView(float* _arr, float* _arrc): 
      arr(_arr), arrc(_arrc){}

    CtensorView(float* _arr, float* _arrc, const int _n0, const int _s0, const int _dev=0): 
      arr(_arr), arrc(_arrc), dims(_n0), strides(_s0), dev(_dev){}

    CtensorView(float* _arr, float* _arrc, const int _n0, const int _n1, const int _s0, const int _s1, const int _dev=0): 
      arr(_arr), arrc(_arrc), dims({_n0,_n1}), strides({_s0,_s1}), dev(_dev){}
    
    CtensorView(float* _arr, float* _arrc, const int _n0, const int _n1, const int _n2, 
      const int _s0, const int _s1, const int _s2, const int _dev=0): 
      arr(_arr), arrc(_arrc), dims({_n0,_n1,_n2}), strides({_s0,_s1,_s2}), dev(_dev){}

    CtensorView(float* _arr, float* _arrc, const int _n0, const int _n1, const int _n2, const int _n3,
      const int _s0, const int _s1, const int _s2, const int _s3, const int _dev=0): 
      arr(_arr), arrc(_arrc), dims({_n0,_n1,_n2,_n3}), strides({_s0,_s1,_s2,_s3}), dev(_dev){}

    CtensorView(float* _arr,  float* _arrc, const Gdims& _dims, const Gstrides& _strides, const int _dev=0):
      arr(_arr), arrc(_arrc), dims(_dims), strides(_strides), dev(_dev){
    }


  public: // ---- Conversions -------------------------------------------------------------------------------


    RtensorView as_real() const{
      CNINE_ASSRT(arrc==arr+1);
      return RtensorView(arr,dims.append(2),strides.append(1),dev);
    }


  public: // ---- Getters ------------------------------------------------------------------------------------


    int ndims() const{
      return dims.size();
    }

    complex<float> operator()(const int i0) const{
      CNINE_CHECK_RANGE(if(i0<0 || i0>=dims[0]) 
	  throw std::out_of_range("cnine::CtensorView: index "+Gindex({i0}).str()+" out of range of view size "+dims.str()));
      return std::complex<float>(arr[strides.offs(i0)],arrc[strides.offs(i0)]);
    }

    complex<float> operator()(const int i0, const int i1) const{
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i0>=dims[0] || i1>=dims[1]) 
	  throw std::out_of_range("cnine::CtensorView: index "+Gindex({i0,i1}).str()+" out of range of view size "+dims.str()));
      return std::complex<float>(arr[strides.offs(i0,i1)],arrc[strides.offs(i0,i1)]);

    }

    complex<float> operator()(const int i0, const int i1, const int i2) const{
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i2<0 || i0>=dims[0] || i1>=dims[1] || i2>=dims[2]) 
	  throw std::out_of_range("cnine::CtensorView: index "+Gindex({i0,i1,i2}).str()+" out of range of view size "+dims.str()));
      return std::complex<float>(arr[strides.offs(i0,i1,i2)],arrc[strides.offs(i0,i1,i2)]);

    }

    complex<float> operator()(const int i0, const int i1, const int i2, const int i3) const{
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i2<0 || i3<0 || i0>=dims[0] || i1>=dims[1] || i2>=dims[2] || i3>=dims[3]) 
	  throw std::out_of_range("cnine::CtensorView: index "+Gindex({i0,i1,i2,i3}).str()+" out of range of view size "+dims.str()));
      return std::complex<float>(arr[strides.offs(i0,i1,i2,i3)],arrc[strides.offs(i0,i1,i2,i3)]);
    }

    complex<float> operator()(const Gindex& ix) const{
      CNINE_CHECK_RANGE(ix.check_range(dims));
      int t=ix(strides);
      return complex<float>(arr[t],arrc[t]);
    }


  public: // ---- Setters ------------------------------------------------------------------------------------


    void set(const int i0, complex<float> x){
      CNINE_CHECK_RANGE(if(i0<0 || i0>=dims[0]) 
	  throw std::out_of_range("cnine::CtensorView: index "+Gindex({i0}).str()+" out of range of view size "+dims.str()));
      arr[strides.offs(i0)]=std::real(x);
      arrc[strides.offs(i0)]=std::imag(x);
    }

    void set(const int i0, const int i1, complex<float> x){
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i0>=dims[0] || i1>=dims[1]) 
	  throw std::out_of_range("cnine::CtensorView: index "+Gindex({i0,i1}).str()+" out of range of view size "+dims.str()));
      arr[strides.offs(i0,i1)]=std::real(x);
      arrc[strides.offs(i0,i1)]=std::imag(x);
    }

    void set(const int i0, const int i1, const int i2, complex<float> x){
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i2<0 || i0>=dims[0] || i1>=dims[1] || i2>=dims[2]) 
	  throw std::out_of_range("cnine::CtensorView: index "+Gindex({i0,i1,i2}).str()+" out of range of view size "+dims.str()));
      arr[strides.offs(i0,i1,i2)]=std::real(x);
      arrc[strides.offs(i0,i1,i2)]=std::imag(x);
    }

    void set(const int i0, const int i1, const int i2, const int i3, complex<float> x){
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i2<0 || i3<0 || i0>=dims[0] || i1>=dims[1] || i2>=dims[2] || i3>=dims[3]) 
	  throw std::out_of_range("cnine::CtensorView: index "+Gindex({i0,i1,i2,i3}).str()+" out of range of view size "+dims.str()));
      arr[strides.offs(i0,i1,i2)]=std::real(x);
      arrc[strides.offs(i0,i1,i2)]=std::imag(x);
    }


  public: // ---- Incrementers -------------------------------------------------------------------------------


    void inc(const int i0, complex<float> x){
      CNINE_CHECK_RANGE(if(i0<0 || i0>=dims[0]) 
	  throw std::out_of_range("cnine::CtensorView: index "+Gindex({i0}).str()+" out of range of view size "+dims.str()));
      arr[strides.offs(i0)]+=std::real(x);
      arrc[strides.offs(i0)]+=std::imag(x);
    }

    void inc(const int i0, const int i1, complex<float> x){
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i0>=dims[0] || i1>=dims[1]) 
	  throw std::out_of_range("cnine::CtensorView: index "+Gindex({i0,i1}).str()+" out of range of view size "+dims.str()));
      arr[strides.offs(i0,i1)]+=std::real(x);
      arrc[strides.offs(i0,i1)]+=std::imag(x);
    }

    void inc(const int i0, const int i1, const int i2, complex<float> x){
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i2<0 || i0>=dims[0] || i1>=dims[1] || i2>=dims[2]) 
	  throw std::out_of_range("cnine::CtensorView: index "+Gindex({i0,i1,i2}).str()+" out of range of view size "+dims.str()));
      arr[strides.offs(i0,i1,i2)]+=std::real(x);
      arrc[strides.offs(i0,i1,i2)]+=std::imag(x);
    }

    void inc(const int i0, const int i1, const int i2, const int i3, complex<float> x){
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i2<0 || i3<0 || i0>=dims[0] || i1>=dims[1] || i2>=dims[2] || i3>=dims[3]) 
	  throw std::out_of_range("cnine::CtensorView: index "+Gindex({i0,i1,i2,i3}).str()+" out of range of view size "+dims.str()));
      arr[strides.offs(i0,i1,i2,i3)]+=std::real(x);
      arrc[strides.offs(i0,i1,i2,i3)]+=std::imag(x);
    }




    //CtensorView block(const int i0, const int i1, const int i2, const int i3, const int m0, const int m1, const int m2, const int m3) const{
    //return CtensorView(arr+i0*s0+i1*s1+i2*s2+i3*s3,m0,m1,m2,m3,s0,s1,s2,s3,dev);
    //}


  public: // ---- Cumulative operations ---------------------------------------------------------------------


  public: // ---- Slices --------------------------------------------------------------------------------------


    CtensorView slice0(const int i){
      CNINE_NDIMS_LEAST(1);
      CNINE_CHECK_RANGE(if(i<0 || i>=dims[0]) 
	  throw std::out_of_range("cnine::CtensorView:slice0(int): index "+to_string(i)+" out of range of [0,"+to_string(dims[0]-1)+"]");)
	return CtensorView(arr+i*dims[0],arrc+i*dims[0],dims.remove(0),strides.remove(0),dev);
    }

    CtensorView slice1(const int i){
      CNINE_NDIMS_LEAST(2);
      CNINE_CHECK_RANGE(if(i<0 || i>=dims[1]) 
	  throw std::out_of_range("cnine::CtensorView:slice1(int): index "+to_string(i)+" out of range of [0,"+to_string(dims[1]-1)+"]");)
	return CtensorView(arr+i*dims[1],arrc+i*dims[1],dims.remove(1),strides.remove(1),dev);
    }

    CtensorView slice2(const int i){
      CNINE_NDIMS_LEAST(3);
      CNINE_CHECK_RANGE(if(i<0 || i>=dims[2]) 
	  throw std::out_of_range("cnine::CtensorView:slice2(int): index "+to_string(i)+" out of range of [0,"+to_string(dims[2]-1)+"]");)
	return CtensorView(arr+i*dims[2],arrc+i*dims[2],dims.remove(2),strides.remove(2),dev);
    }

    CtensorView slice3(const int i){
      CNINE_NDIMS_LEAST(4);
      CNINE_CHECK_RANGE(if(i<0 || i>=dims[3]) 
	  throw std::out_of_range("cnine::CtensorView:slice3(int): index "+to_string(i)+" out of range of [0,"+to_string(dims[3]-1)+"]");)
	return CtensorView(arr+i*dims[3],arrc+i*dims[3],dims.remove(3),strides.remove(3),dev);
    }


  public: // ---- Other views -------------------------------------------------------------------------------


    Ctensor2_view fuse0X() const{
      if(dims.size()==0) return Ctensor2_view(arr,arrc,1,1,0,0,dev);
      if(dims.size()==1) return Ctensor2_view(arr,arrc,dims[0],1,strides[0],0,dev);
      return Ctensor2_view(arr,arrc,dims[0],strides[0]/strides.back(),strides[0],strides.back(),dev);
    }    


  public: // ---- Conversions -------------------------------------------------------------------------------


    Gtensor<complex<float> > gtensor() const{
      assert(dev==0);
      Gtensor<complex<float> > R(dims,fill::raw);
      for(int i=0; i<R.asize; i++)
	R.arr[i]=(*this)(Gindex(i,strides));
      return R;
    }
    
    
  public: // ---- I/O ----------------------------------------------------------------------------------------

  
    string str(const string indent="") const{
      return gtensor().str(indent);
    }

    friend ostream& operator<<(ostream& stream, const CtensorView& x){
      stream<<x.str(); return stream;
    }


  };


}


#endif 
