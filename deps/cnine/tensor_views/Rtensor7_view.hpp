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


#ifndef _CnineRtensor7_view
#define _CnineRtensor7_view

#include "Cnine_base.hpp"
#include "Gstrides.hpp"
#include "Gtensor.hpp"

#include "Rtensor6_view.hpp"


namespace cnine{


  class Rtensor7_view{
  public:

    float* arr;
    int n0,n1,n2,n3,n4,n5,n6;
    int s0,s1,s2,s3,s4,s5,s6;
    int dev=0;

  public:

    Rtensor7_view(){}

    Rtensor7_view(float* _arr): 
      arr(_arr){}

    Rtensor7_view(float* _arr, const int _n0, const int _n1, const int _n2, const int _n3, const int _n4, const int _n5, const int _n6, 
      const int _s0, const int _s1, const int _s2, const int _s3, const int _s4, const int _s5, const int _s6, const int _dev=0): 
      arr(_arr), n0(_n0), n1(_n1), n2(_n2), n3(_n3), n4(_n4), n5(_n5), n6(_n6),
      s0(_s0), s1(_s1), s2(_s2), s3(_s3), s4(_s4), s5(_s5), s6(_s6), dev(_dev){}

    Rtensor7_view(float* _arr,  const Gdims& _dims, const Gstrides& _strides, const int _dev=0):
      arr(_arr), dev(_dev){
      CNINE_ASSRT(_dims.size()==6);
      n0=_dims[0];
      n1=_dims[1];
      n2=_dims[2];
      n3=_dims[3];
      n4=_dims[4];
      n5=_dims[5];
      n6=_dims[6];
      s0=_strides[0];
      s1=_strides[1];
      s2=_strides[2];
      s3=_strides[3];
      s4=_strides[4];
      s5=_strides[5];
      s6=_strides[6];
    }



  public: // ---- Access ------------------------------------------------------------------------------------


    Gdims get_dims() const{
      return Gdims({n0,n1,n2,n3,n4,n5,n6});
    }

    float operator()(const int i0, const int i1, const int i2, const int i3, const int i4, const int i5, const int i6) const{
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i2<0 || i3<0 || i4<0 || i5<0 || i6<0 || i0>=n0 || i1>=n1 || i2>=n2 || i3>=n3 || i4>=n4 || i5>=n5 || i6>=n6) 
	  throw std::out_of_range("cnine::Rtensor7_view: index "+Gindex({i0,i1,i2,i3,i4,i5,i6}).str()+" out of range of view size "+Gdims({n0,n1,n2,n3,n4,n5,n6}).str()));
      return arr[s0*i0+s1*i1+s2*i2+s3*i3+i4*s4+i5*s5+i6*s6];
    }

    void set(const int i0, const int i1, const int i2, const int i3, const int i4, const int i5, const int i6, float x) const{
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i2<0 || i3<0 || i4<0 || i5<0 || i6<0 || i0>=n0 || i1>=n1 || i2>=n2 || i3>=n3 || i4>=n4 || i5>=n5 || i6>=n6) 
	  throw std::out_of_range("cnine::Rtensor7_view: index "+Gindex({i0,i1,i2,i3,i4,i5,i6}).str()+" out of range of view size "+Gdims({n0,n1,n2,n3,n4,n5,n6}).str()));
      arr[s0*i0+s1*i1+s2*i2+s3*i3+i4*s4+i5*s5+i6*s6]=x;
    }

    void inc(const int i0, const int i1, const int i2, const int i3, const int i4, const int i5, const int i6, float x) const{
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i2<0 || i3<0 || i4<0 || i5<0 || i6<0 || i0>=n0 || i1>=n1 || i2>=n2 || i3>=n3 || i4>=n4 || i5>=n5 || i6>=n6) 
	  throw std::out_of_range("cnine::Rtensor7_view: index "+Gindex({i0,i1,i2,i3,i4,i5,i6}).str()+" out of range of view size "+Gdims({n0,n1,n2,n3,n4,n5,n6}).str()));
      arr[s0*i0+s1*i1+s2*i2+s3*i3+i4*s4+i5*s5+i6*s6]+=x;
    }

    Rtensor7_view block(const int i0, const int i1, const int i2, const int i3, const int i4, const int i5, const int i6, 
      const int m0, const int m1, const int m2, const int m3, const int m4, const int m5, const int m6) const{
      return Rtensor7_view(arr+i0*s0+i1*s1+i2*s2+i3*s3+i4*s4+i5*s5+i6*s6,m0,m1,m2,m3,m4,m5,m6,s0,s1,s2,s3,s4,s5,s6,dev);
    }


  public: // ---- Cumulative operations ---------------------------------------------------------------------

    
    void add(const Rtensor7_view& y) const{
      CNINE_ASSRT(y.n0==n0);
      CNINE_ASSRT(y.n1==n1);
      CNINE_ASSRT(y.n2==n2);
      CNINE_ASSRT(y.n3==n3);
      CNINE_ASSRT(y.n4==n4);
      CNINE_ASSRT(y.n5==n5);
      CNINE_ASSRT(y.n6==n6);
      for(int i0=0; i0<n0; i0++)
	for(int i1=0; i1<n1; i1++)
	  for(int i2=0; i2<n2; i2++)
	    for(int i3=0; i3<n3; i3++)
	      for(int i4=0; i4<n4; i4++)
		for(int i5=0; i5<n5; i5++)
		  for(int i6=0; i6<n6; i6++)
		    inc(i0,i1,i2,i3,i4,i5,i6,y(i0,i1,i2,i3,i4,i5,i6));
    }


  public: // ---- Reductions --------------------------------------------------------------------------------


    void reduce0_destructively_into(const Rtensor6_view& r) const{
      reduce0_destructively();
      r.add(slice0(0));
    }

    void reduce1_destructively_into(const Rtensor6_view& r) const{
      reduce1_destructively();
      r.add(slice1(0));
    }

    void reduce2_destructively_into(const Rtensor6_view& r) const{
      reduce2_destructively();
      r.add(slice2(0));
    }

    void reduce3_destructively_into(const Rtensor6_view& r) const{
      reduce3_destructively();
      r.add(slice3(0));
    }


    void reduce0_destructively() const{
      fuse12().reduce0_destructively();
    }

    void reduce1_destructively() const{
      fuse23().reduce1_destructively();
    }

    void reduce2_destructively() const{
      fuse01().reduce1_destructively();
    }

    void reduce3_destructively() const{
      fuse01().reduce2_destructively();
    }


  public: // ---- Broadcasting ------------------------------------------------------------------------------

    /*
    void broadcast0(const Rtensor6_view& x){
      CNINE_ASSRT(x.n0==n1);
      CNINE_ASSRT(x.n1==n2);
      CNINE_ASSRT(x.n2==n3);
      CNINE_ASSRT(x.n3==n4);
      CNINE_ASSRT(x.n4==n5);
      fuse45().broadcast0(x.fuse34());
    }

    void broadcast1(const Rtensor6_view& x){
      CNINE_ASSRT(x.n0==n0);
      CNINE_ASSRT(x.n1==n2);
      CNINE_ASSRT(x.n2==n3);
      CNINE_ASSRT(x.n3==n4);
      CNINE_ASSRT(x.n4==n5);
      fuse45().broadcast1(x.fuse34());
    }

    void broadcast2(const Rtensor6_view& x){
      CNINE_ASSRT(x.n0==n0);
      CNINE_ASSRT(x.n1==n1);
      CNINE_ASSRT(x.n2==n3);
      CNINE_ASSRT(x.n3==n4);
      CNINE_ASSRT(x.n4==n5);
      fuse01().broadcast1(x.fuse01());
    }

    void broadcast3(const Rtensor6_view& x){
      CNINE_ASSRT(x.n0==n0);
      CNINE_ASSRT(x.n1==n1);
      CNINE_ASSRT(x.n2==n2);
      CNINE_ASSRT(x.n3==n4);
      CNINE_ASSRT(x.n4==n5);
      fuse01().broadcast2(x.fuse01());
    }

    void broadcast4(const Rtensor6_view& x){
      CNINE_ASSRT(x.n0==n0);
      CNINE_ASSRT(x.n1==n1);
      CNINE_ASSRT(x.n2==n2);
      CNINE_ASSRT(x.n3==n3);
      CNINE_ASSRT(x.n4==n5);
      fuse01().broadcast3(x.fuse01());
    }

    void broadcast5(const Rtensor6_view& x){
      CNINE_ASSRT(x.n0==n0);
      CNINE_ASSRT(x.n1==n1);
      CNINE_ASSRT(x.n2==n2);
      CNINE_ASSRT(x.n3==n3);
      CNINE_ASSRT(x.n4==n4);
      fuse01().broadcast4(x.fuse01());
    }
    */

  public: // ---- Other views -------------------------------------------------------------------------------


    Rtensor6_view slice0(const int i) const{
      CNINE_CHECK_RANGE(if(i<0 || i>=n0) 
	  throw std::out_of_range("cnine::Rtensor7_view:slice0(int): index "+to_string(i)+" out of range of [0,"+to_string(n0-1)+"]");)
	return Rtensor6_view(arr+i*s0,n1,n2,n3,n4,n5,n6,s1,s2,s3,s4,s5,s6,dev);
    }

    Rtensor6_view slice1(const int i) const{
      CNINE_CHECK_RANGE(if(i<0 || i>=n1) 
	  throw std::out_of_range("cnine::Rtensor7_view:slice1(int): index "+to_string(i)+" out of range of [0,"+to_string(n1-1)+"]");)
	return Rtensor6_view(arr+i*s1,n0,n2,n3,n4,n5,n6,s0,s2,s3,s4,s5,s6,dev);
    }

    Rtensor6_view slice2(const int i) const{
      CNINE_CHECK_RANGE(if(i<0 || i>=n2) 
	  throw std::out_of_range("cnine::Rtensor7_view:slice2(int): index "+to_string(i)+" out of range of [0,"+to_string(n2-1)+"]");)
	return Rtensor6_view(arr+i*s2,n0,n1,n3,n4,n5,n6,s0,s1,s3,s4,s5,s6,dev);
    }

    Rtensor6_view slice3(const int i) const{
      CNINE_CHECK_RANGE(if(i<0 || i>=n3) 
	  throw std::out_of_range("cnine::Rtensor7_view:slice3(int): index "+to_string(i)+" out of range of [0,"+to_string(n3-1)+"]");)
	return Rtensor6_view(arr+i*s3,n0,n1,n2,n4,n5,n6,s0,s1,s2,s4,s5,s6,dev);
    }

    Rtensor6_view slice4(const int i) const{
      CNINE_CHECK_RANGE(if(i<0 || i>=n4) 
	  throw std::out_of_range("cnine::Rtensor7_view:slice4(int): index "+to_string(i)+" out of range of [0,"+to_string(n4-1)+"]");)
	return Rtensor6_view(arr+i*s4,n0,n1,n2,n3,n5,n6,s0,s1,s2,s3,s5,s6,dev);
    }

    Rtensor6_view slice5(const int i) const{
      CNINE_CHECK_RANGE(if(i<0 || i>=n5) 
	  throw std::out_of_range("cnine::Rtensor7_view:slice5(int): index "+to_string(i)+" out of range of [0,"+to_string(n5-1)+"]");)
	return Rtensor6_view(arr+i*s5,n0,n1,n2,n3,n4,n6,s0,s1,s2,s3,s4,s6,dev);
    }

    Rtensor6_view slice6(const int i) const{
      CNINE_CHECK_RANGE(if(i<0 || i>=n6) 
	  throw std::out_of_range("cnine::Rtensor7_view:slice5(int): index "+to_string(i)+" out of range of [0,"+to_string(n5-1)+"]");)
	return Rtensor6_view(arr+i*s6,n0,n1,n2,n3,n4,n5,s0,s1,s2,s3,s4,s5,dev);
    }

    Rtensor6_view fuse01() const{
      return Rtensor6_view(arr,n0*n1,n2,n3,n4,n5,n6,s1,s2,s3,s4,s5,s6,dev);
    }    

    Rtensor6_view fuse12() const{
      return Rtensor6_view(arr,n0,n1*n2,n3,n4,n5,n6,s0,s2,s3,s4,s5,s6,dev);
    }    

    Rtensor6_view fuse23() const{
      return Rtensor6_view(arr,n0,n1,n2*n3,n4,n5,n6,s0,s1,s3,s4,s5,s6,dev);
    }    

    Rtensor6_view fuse34() const{
      return Rtensor6_view(arr,n0,n1,n2,n3*n4,n5,n6,s0,s1,s2,s4,s5,s6,dev);
    }    

    Rtensor6_view fuse45() const{
      return Rtensor6_view(arr,n0,n1,n2,n3,n4*n5,n6,s0,s1,s2,s3,s5,s6,dev);
    }    

    Rtensor6_view fuse56() const{
      return Rtensor6_view(arr,n0,n1,n2,n3,n4,n5*n6,s0,s1,s2,s3,s4,s6,dev);
    }    

    /*
    Rtensor3_view slice01(const int i, const int j) const{
      CNINE_CHECK_RANGE(if(i<0 || i>=n0) 
	  throw std::out_of_range("cnine::Rtensor7_view:slice01(int): index "+to_string(i)+" out of range of [0,"+to_string(n0-1)+"]");)
      CNINE_CHECK_RANGE(if(j<0 || j>=n1) 
	  throw std::out_of_range("cnine::Rtensor7_view:slice01(int): index "+to_string(j)+" out of range of [0,"+to_string(n1-1)+"]");)
	return Rtensor3_view(arr+i*s0+j*s1,n2,n3,n4,s2,s3,s4,dev);
    }

    Rtensor3_view slice02(const int i, const int j) const{
      CNINE_CHECK_RANGE(if(i<0 || i>=n0) 
	  throw std::out_of_range("cnine::Rtensor7_view:slice02(int): index "+to_string(i)+" out of range of [0,"+to_string(n0-1)+"]");)
      CNINE_CHECK_RANGE(if(j<0 || j>=n2) 
	  throw std::out_of_range("cnine::Rtensor7_view:slice02(int): index "+to_string(j)+" out of range of [0,"+to_string(n2-1)+"]");)
	return Rtensor3_view(arr+i*s0+j*s2,n1,n3,n4,s1,s3,s4,dev);
    }

    Rtensor3_view slice03(const int i, const int j) const{
      CNINE_CHECK_RANGE(if(i<0 || i>=n0) 
	  throw std::out_of_range("cnine::Rtensor7_view:slice03(int): index "+to_string(i)+" out of range of [0,"+to_string(n0-1)+"]");)
      CNINE_CHECK_RANGE(if(j<0 || j>=n3) 
	  throw std::out_of_range("cnine::Rtensor7_view:slice03(int): index "+to_string(j)+" out of range of [0,"+to_string(n3-1)+"]");)
	return Rtensor3_view(arr+i*s0+j*s3,n1,n2,n4,s1,s2,s4,dev);
    }

    Rtensor3_view slice04(const int i, const int j) const{
      CNINE_CHECK_RANGE(if(i<0 || i>=n0) 
	  throw std::out_of_range("cnine::Rtensor7_view:slice04(int): index "+to_string(i)+" out of range of [0,"+to_string(n0-1)+"]");)
      CNINE_CHECK_RANGE(if(j<0 || j>=n4) 
	  throw std::out_of_range("cnine::Rtensor7_view:slice04(int): index "+to_string(j)+" out of range of [0,"+to_string(n4-1)+"]");)
	return Rtensor3_view(arr+i*s0+j*s4,n1,n2,n3,s1,s2,s3,dev);
    }

    Rtensor3_view slice12(const int i, const int j) const{
      CNINE_CHECK_RANGE(if(i<0 || i>=n1) 
	  throw std::out_of_range("cnine::Rtensor7_view:slice12(int): index "+to_string(i)+" out of range of [0,"+to_string(n1-1)+"]");)
      CNINE_CHECK_RANGE(if(j<0 || j>=n2) 
	  throw std::out_of_range("cnine::Rtensor7_view:slice12(int): index "+to_string(j)+" out of range of [0,"+to_string(n2-1)+"]");)
	return Rtensor3_view(arr+i*s1+j*s2,n0,n3,n4,s0,s3,s4,dev);
    }

    Rtensor3_view slice13(const int i, const int j) const{
      CNINE_CHECK_RANGE(if(i<0 || i>=n1) 
	  throw std::out_of_range("cnine::Rtensor7_view:slice13(int): index "+to_string(i)+" out of range of [0,"+to_string(n1-1)+"]");)
      CNINE_CHECK_RANGE(if(j<0 || j>=n3) 
	  throw std::out_of_range("cnine::Rtensor7_view:slice13(int): index "+to_string(j)+" out of range of [0,"+to_string(n3-1)+"]");)
	return Rtensor3_view(arr+i*s1+j*s3,n0,n2,n4,s0,s2,s4,dev);
    }

    Rtensor3_view slice14(const int i, const int j) const{
      CNINE_CHECK_RANGE(if(i<0 || i>=n1) 
	  throw std::out_of_range("cnine::Rtensor7_view:slice14(int): index "+to_string(i)+" out of range of [0,"+to_string(n1-1)+"]");)
      CNINE_CHECK_RANGE(if(j<0 || j>=n4) 
	  throw std::out_of_range("cnine::Rtensor7_view:slice14(int): index "+to_string(j)+" out of range of [0,"+to_string(n4-1)+"]");)
	return Rtensor3_view(arr+i*s1+j*s4,n0,n2,n3,s0,s2,s3,dev);
    }

    Rtensor3_view slice23(const int i, const int j) const{
      CNINE_CHECK_RANGE(if(i<0 || i>=n2) 
	  throw std::out_of_range("cnine::Rtensor7_view:slice23(int): index "+to_string(i)+" out of range of [0,"+to_string(n2-1)+"]");)
      CNINE_CHECK_RANGE(if(j<0 || j>=n3) 
	  throw std::out_of_range("cnine::Rtensor7_view:slice23(int): index "+to_string(j)+" out of range of [0,"+to_string(n3-1)+"]");)
	return Rtensor3_view(arr+i*s2+j*s3,n0,n1,n4,s0,s1,s4,dev);
    }

    Rtensor3_view slice24(const int i, const int j) const{
      CNINE_CHECK_RANGE(if(i<0 || i>=n2) 
	  throw std::out_of_range("cnine::Rtensor7_view:slice24(int): index "+to_string(i)+" out of range of [0,"+to_string(n2-1)+"]");)
      CNINE_CHECK_RANGE(if(j<0 || j>=n4) 
	  throw std::out_of_range("cnine::Rtensor7_view:slice24(int): index "+to_string(j)+" out of range of [0,"+to_string(n4-1)+"]");)
	return Rtensor3_view(arr+i*s2+j*s4,n0,n1,n3,s0,s1,s3,dev);
    }

    Rtensor3_view slice34(const int i, const int j) const{
      CNINE_CHECK_RANGE(if(i<0 || i>=n3) 
	  throw std::out_of_range("cnine::Rtensor7_view:slice34(int): index "+to_string(i)+" out of range of [0,"+to_string(n3-1)+"]");)
      CNINE_CHECK_RANGE(if(j<0 || j>=n4) 
	  throw std::out_of_range("cnine::Rtensor7_view:slice34(int): index "+to_string(j)+" out of range of [0,"+to_string(n4-1)+"]");)
	return Rtensor3_view(arr+i*s3+j*s4,n0,n1,n2,s0,s1,s2,dev);
    }
    */


  public: // ---- Conversions -------------------------------------------------------------------------------

    
    Gtensor<float> gtensor() const{
      Gtensor<float> R({n0,n1,n2,n3,n4,n5},fill::raw);
      for(int i0=0; i0<n0; i0++)
	for(int i1=0; i1<n1; i1++)
	  for(int i2=0; i2<n2; i2++)
	    for(int i3=0; i3<n3; i3++)
	      for(int i4=0; i4<n4; i4++)
		for(int i5=0; i5<n5; i5++)
		  for(int i6=0; i6<n6; i6++)
		    R(i0,i1,i2,i3,i4,i5,i6)=(*this)(i0,i1,i2,i3,i4,i5,i6);
      return R;
    }
    
    
  public: // ---- I/O ----------------------------------------------------------------------------------------

  
    string str(const string indent="") const{
      return gtensor().str(indent);
    }

    friend ostream& operator<<(ostream& stream, const Rtensor7_view& x){
      stream<<x.str(); return stream;
    }


  };

  
  inline Rtensor7_view split0(const Rtensor6_view& x, const int i, const int j){
    CNINE_ASSRT(i*j==x.n0);
    return Rtensor7_view(x.arr,i,j,x.n1,x.n2,x.n3,x.n4,x.n5,x.s0*j,x.s0,x.s1,x.s2,x.s3,x.s4,x.s5,x.dev);
  }

  inline Rtensor7_view split1(const Rtensor6_view& x, const int i, const int j){
    CNINE_ASSRT(i*j==x.n1);
    return Rtensor7_view(x.arr,x.n0,i,j,x.n2,x.n3,x.n4,x.n5,x.s0,x.s1*j,x.s1,x.s2,x.s3,x.s4,x.s5,x.dev);
    }

  inline Rtensor7_view split2(const Rtensor6_view& x, const int i, const int j){
    CNINE_ASSRT(i*j==x.n2);
    return Rtensor7_view(x.arr,x.n0,x.n1,i,j,x.n3,x.n4,x.n5,x.s0,x.s1,x.s2*j,x.s2,x.s3,x.s4,x.s5,x.dev);
  }
 
  inline Rtensor7_view split3(const Rtensor6_view& x, const int i, const int j){
    CNINE_ASSRT(i*j==x.n3);
    return Rtensor7_view(x.arr,x.n0,x.n1,x.n2,i,j,x.n4,x.n5,x.s0,x.s1,x.s2,x.s3*j,x.s3,x.s4,x.s5,x.dev);
  }
 
  inline Rtensor7_view split4(const Rtensor6_view& x, const int i, const int j){
    CNINE_ASSRT(i*j==x.n4);
    return Rtensor7_view(x.arr,x.n0,x.n1,x.n2,x.n3,i,j,x.n5,x.s0,x.s1,x.s2,x.s3,x.s4*j,x.s4,x.s5,x.dev);
  }
 
  inline Rtensor7_view split5(const Rtensor6_view& x, const int i, const int j){
    CNINE_ASSRT(i*j==x.n5);
    return Rtensor7_view(x.arr,x.n0,x.n1,x.n2,x.n3,x.n4,i,j,x.s0,x.s1,x.s2,x.s3,x.s4,x.s5*j,x.s5,x.dev);
  }
 

}


#endif 
