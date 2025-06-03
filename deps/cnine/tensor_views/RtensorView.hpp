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


#ifndef _CnineRtensorView
#define _CnineRtensorView

#include "Cnine_base.hpp"
#include "Gstrides.hpp"
#include "Gtensor.hpp"

#include "Rtensor1_view.hpp"
#include "Rtensor2_view.hpp"
#include "Rtensor3_view.hpp"
#include "Rtensor4_view.hpp"
#include "Rtensor5_view.hpp"
#include "Rtensor6_view.hpp"


namespace cnine{


  class RtensorView{
  public:

    float* arr;
    const Gdims dims;
    const Gstrides strides;
    int dev=0;

  public:

    RtensorView(){}

    RtensorView(float* _arr): 
      arr(_arr){}

    RtensorView(float* _arr, const int _n0, const int _s0, const int _dev=0): 
      arr(_arr), dims(_n0), strides(_s0), dev(_dev){}

    RtensorView(float* _arr, const int _n0, const int _n1, const int _s0, const int _s1, const int _dev=0): 
      arr(_arr), dims({_n0,_n1}), strides({_s0,_s1}), dev(_dev){}

    RtensorView(float* _arr, const int _n0, const int _n1, const int _n2, 
      const int _s0, const int _s1, const int _s2, const int _dev=0): 
      arr(_arr), dims({_n0,_n1,_n2}), strides({_s0,_s1,_s2}), dev(_dev){}

    RtensorView(float* _arr, const int _n0, const int _n1, const int _n2, const int _n3,
      const int _s0, const int _s1, const int _s2, const int _s3, const int _dev=0): 
      arr(_arr), dims({_n0,_n1,_n2,_n3}), strides({_s0,_s1,_s2,_s3}), dev(_dev){}

    RtensorView(float* _arr,  const Gdims& _dims, const Gstrides& _strides, const int _dev=0):
      arr(_arr), dims(_dims), strides(_strides), dev(_dev){
    }


  public: // ---- Copying ------------------------------------------------------------------------------------


    //RtensorView(const RtensorView& x):
    //arr(x.arr), dims(x.dims), strides(x.strides), dev(x.dev){}


  public: // ---- Getters ------------------------------------------------------------------------------------

    
    int ndims() const{
      return dims.size();
    }

    float operator()(const int i0) const{
      CNINE_CHECK_RANGE(if(i0<0 || i0>=dims[0]) 
	  throw std::out_of_range("cnine::RtensorView: index "+Gindex({i0}).str()+" out of range of view size "+dims.str()));
      return arr[strides.offs(i0)];
    }

    float operator()(const int i0, const int i1) const{
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i0>=dims[0] || i1>=dims[1]) 
	  throw std::out_of_range("cnine::RtensorView: index "+Gindex({i0,i1}).str()+" out of range of view size "+dims.str()));
      return arr[strides.offs(i0,i1)];
    }

    float operator()(const int i0, const int i1, const int i2) const{
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i2<0 || i0>=dims[0] || i1>=dims[1] || i2>=dims[2]) 
	  throw std::out_of_range("cnine::RtensorView: index "+Gindex({i0,i1,i2}).str()+" out of range of view size "+dims.str()));
      return arr[strides.offs(i0,i1,i2)];
    }

    float operator()(const int i0, const int i1, const int i2, const int i3) const{
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i2<0 || i3<0 || i0>=dims[0] || i1>=dims[1] || i2>=dims[2] || i3>=dims[3]) 
	  throw std::out_of_range("cnine::RtensorView: index "+Gindex({i0,i1,i2,i3}).str()+" out of range of view size "+dims.str()));
      return arr[strides.offs(i0,i1,i2,i3)];
    }


  public: // ---- Setters ------------------------------------------------------------------------------------


    void set(const int i0, float x){
      CNINE_CHECK_RANGE(if(i0<0 || i0>=dims[0]) 
	  throw std::out_of_range("cnine::RtensorView: index "+Gindex({i0}).str()+" out of range of view size "+dims.str()));
      arr[strides.offs(i0)]=x;
    }

    void set(const int i0, const int i1, float x){
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i0>=dims[0] || i1>=dims[1]) 
	  throw std::out_of_range("cnine::RtensorView: index "+Gindex({i0,i1}).str()+" out of range of view size "+dims.str()));
      arr[strides.offs(i0,i1)]=x;
    }

    void set(const int i0, const int i1, const int i2, float x){
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i2<0 || i0>=dims[0] || i1>=dims[1] || i2>=dims[2]) 
	  throw std::out_of_range("cnine::RtensorView: index "+Gindex({i0,i1,i2}).str()+" out of range of view size "+dims.str()));
      arr[strides.offs(i0,i1,i2)]=x;
    }

    void set(const int i0, const int i1, const int i2, const int i3, float x){
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i2<0 || i3<0 || i0>=dims[0] || i1>=dims[1] || i2>=dims[2] || i3>=dims[3]) 
	  throw std::out_of_range("cnine::RtensorView: index "+Gindex({i0,i1,i2,i3}).str()+" out of range of view size "+dims.str()));
      arr[strides.offs(i0,i1,i2,i3)]=x;
    }


  public: // ---- Incrementers -------------------------------------------------------------------------------


    void inc(const int i0, float x){
      CNINE_CHECK_RANGE(if(i0<0 || i0>=dims[0]) 
	  throw std::out_of_range("cnine::RtensorView: index "+Gindex({i0}).str()+" out of range of view size "+dims.str()));
      arr[strides.offs(i0)]+=x;
    }

    void inc(const int i0, const int i1, float x){
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i0>=dims[0] || i1>=dims[1]) 
	  throw std::out_of_range("cnine::RtensorView: index "+Gindex({i0,i1}).str()+" out of range of view size "+dims.str()));
      arr[strides.offs(i0,i1)]+=x;
    }

    void inc(const int i0, const int i1, const int i2, float x){
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i2<0 || i0>=dims[0] || i1>=dims[1] || i2>=dims[2]) 
	  throw std::out_of_range("cnine::RtensorView: index "+Gindex({i0,i1,i2}).str()+" out of range of view size "+dims.str()));
      arr[strides.offs(i0,i1,i2)]+=x;
    }

    void inc(const int i0, const int i1, const int i2, const int i3, float x){
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i2<0 || i3<0 || i0>=dims[0] || i1>=dims[1] || i2>=dims[2] || i3>=dims[3]) 
	  throw std::out_of_range("cnine::RtensorView: index "+Gindex({i0,i1,i2,i3}).str()+" out of range of view size "+dims.str()));
      arr[strides.offs(i0,i1,i2,i3)]+=x;
    }




    //RtensorView block(const int i0, const int i1, const int i2, const int i3, const int m0, const int m1, const int m2, const int m3) const{
    //return RtensorView(arr+i0*s0+i1*s1+i2*s2+i3*s3,m0,m1,m2,m3,s0,s1,s2,s3,dev);
    //}


  public: // ---- Cumulative operations ---------------------------------------------------------------------

    /*
    void add(const RtensorView& y){
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

  public: // ---- Other views ---------------------------------------------------------------------------------


    Rtensor1_view view1() const{
      CNINE_NDIMS_IS(1);
      return Rtensor1_view(arr,dims[0],strides[0],dev);
    }

    Rtensor2_view view2() const{
      CNINE_NDIMS_IS(2);
      return Rtensor2_view(arr,dims[0],dims[1],strides[0],strides[1],dev);
    }

    Rtensor3_view view3() const{
      CNINE_NDIMS_IS(3);
      return Rtensor3_view(arr,dims[0],dims[1],dims[2],strides[0],strides[1],strides[2],dev);
    }

    Rtensor4_view view4() const{
      CNINE_NDIMS_IS(4);
      return Rtensor4_view(arr,dims[0],dims[1],dims[2],dims[3],strides[0],strides[1],strides[2],strides[3],dev);
    }

    Rtensor5_view view5() const{
      CNINE_NDIMS_IS(5);
      return Rtensor5_view(arr,dims[0],dims[1],dims[2],dims[3],dims[4],strides[0],strides[1],strides[2],strides[3],strides[4],dev);
    }

    Rtensor6_view view6() const{
      CNINE_NDIMS_IS(6);
      return Rtensor6_view(arr,dims[0],dims[1],dims[2],dims[3],dims[4],dims[5],strides[0],strides[1],strides[2],strides[3],strides[4],strides[5],dev);
    }


    Rtensor1_view flatten() const{
      CNINE_NDIMS_IS(1);
      return Rtensor1_view(arr,dims.asize(),strides.back(),dev);
    }

    Rtensor2_view matricize(const int n=1) const{
      CNINE_NDIMS_LEAST(n+1);
      Gdims d({1,1});
      for(int i=0; i<n; i++) d[0]*=dims[i];
      for(int i=n; i<dims.size(); i++) d[1]*=dims[i];
      Gstrides s({strides[n-1],1});
      return Rtensor2_view(arr,d,s,dev);
    }


  public: // ---- Slices --------------------------------------------------------------------------------------


    RtensorView slice0(const int i){
      CNINE_NDIMS_LEAST(1);
      CNINE_CHECK_RANGE(if(i<0 || i>=dims[0]) 
	  throw std::out_of_range("cnine::RtensorView:slice0(int): index "+to_string(i)+" out of range of [0,"+to_string(dims[0]-1)+"]");)
	return RtensorView(arr+i*dims[0],dims.remove(0),strides.remove(0),dev);
    }

    RtensorView slice1(const int i){
      CNINE_NDIMS_LEAST(2);
      CNINE_CHECK_RANGE(if(i<0 || i>=dims[1]) 
	  throw std::out_of_range("cnine::RtensorView:slice1(int): index "+to_string(i)+" out of range of [0,"+to_string(dims[1]-1)+"]");)
	return RtensorView(arr+i*dims[1],dims.remove(1),strides.remove(1),dev);
    }

    RtensorView slice2(const int i){
      CNINE_NDIMS_LEAST(3);
      CNINE_CHECK_RANGE(if(i<0 || i>=dims[2]) 
	  throw std::out_of_range("cnine::RtensorView:slice2(int): index "+to_string(i)+" out of range of [0,"+to_string(dims[2]-1)+"]");)
	return RtensorView(arr+i*dims[2],dims.remove(2),strides.remove(2),dev);
    }

    RtensorView slice3(const int i){
      CNINE_NDIMS_LEAST(4);
      CNINE_CHECK_RANGE(if(i<0 || i>=dims[3]) 
	  throw std::out_of_range("cnine::RtensorView:slice3(int): index "+to_string(i)+" out of range of [0,"+to_string(dims[3]-1)+"]");)
	return RtensorView(arr+i*dims[3],dims.remove(3),strides.remove(3),dev);
    }

    RtensorView slice01(const int i0, const int i1){
      CNINE_NDIMS_LEAST(2);
      CNINE_CHECK_RANGE(if(i0<0 || i0>=dims[0]) 
	  throw std::out_of_range("cnine::RtensorView:slice01(int,int): index "+to_string(i0)+" out of range of [0,"+to_string(dims[0]-1)+"]");)
      CNINE_CHECK_RANGE(if(i1<0 || i1>=dims[1]) 
	  throw std::out_of_range("cnine::RtensorView:slice01(int,int): index "+to_string(i1)+" out of range of [0,"+to_string(dims[1]-1)+"]");)
	return RtensorView(arr+i0*dims[0]+i1*dims[1],dims.chunk(2),strides.chunk(2),dev);
    }

    RtensorView slice012(const int i0, const int i1, const int i2){
      CNINE_NDIMS_LEAST(3);
      CNINE_CHECK_RANGE(if(i0<0 || i0>=dims[0]) 
	  throw std::out_of_range("cnine::RtensorView:slice012(int,int,int): index "+to_string(i0)+" out of range of [0,"+to_string(dims[0]-1)+"]");)
      CNINE_CHECK_RANGE(if(i1<0 || i1>=dims[1]) 
	  throw std::out_of_range("cnine::RtensorView:slice012(int,int,int): index "+to_string(i1)+" out of range of [0,"+to_string(dims[1]-1)+"]");)
      CNINE_CHECK_RANGE(if(i2<0 || i2>=dims[2]) 
	  throw std::out_of_range("cnine::RtensorView:slice012(int,int,int): index "+to_string(i2)+" out of range of [0,"+to_string(dims[2]-1)+"]");)
	return RtensorView(arr+i0*dims[0]+i1*dims[1]+i2*dims[2],dims.chunk(3),strides.chunk(3),dev);
    }


  public: // ---- Fusing -------------------------------------------------------------------------------------


    //fuse i with i+1
    RtensorView fuse(int i) const{
      if(i<0) i=dims.size()-(-i);
      CNINE_NDIMS_LEAST(i+2);
      Gdims _dims=dims.remove(i);
      _dims[i]=dims[i]*dims[i+1];
      Gstrides _strides=strides.remove(i);
      return RtensorView(arr,_dims,_strides,dev);
    }

    RtensorView fuse01() const{
      CNINE_NDIMS_LEAST(2);
      Gdims d=dims.remove(1); d[0]*=dims[1];
      Gstrides s=strides.remove(0);
      return RtensorView(arr,d,s,dev);
    }    

    RtensorView fuse12() const{
      CNINE_NDIMS_LEAST(3);
      Gdims d=dims.remove(2); d[1]*=dims[2];
      Gstrides s=strides.remove(1);
      return RtensorView(arr,d,s,dev);
    }    

    RtensorView fuse23() const{
      CNINE_NDIMS_LEAST(4);
      Gdims d=dims.remove(3); d[2]*=dims[1];
      Gstrides s=strides.remove(2);
      return RtensorView(arr,d,s,dev);
    }    

    Rtensor3_view fuse_all_but_last_two() const{
      CNINE_NDIMS_LEAST(3);
      return Rtensor3_view(arr,dims.fuse(0,dims.size()-2),strides.fuse(0,dims.size()-2),dev);
    }


    /*
    Rtensor3_view fuse12(){
      return Rtensor3_view(arr,n0,n1*n2,n3,s0,s2,s3,dev);
    }    

    Rtensor3_view fuse23(){
      return Rtensor3_view(arr,n0,n1,n2*n3,s0,s1,s3,dev);
    }    
    */
    /*

    Rtensor2_view slice01(const int i, const int j){
      CNINE_CHECK_RANGE(if(i<0 || i>=n0) 
	  throw std::out_of_range("cnine::RtensorView:slice01(int): index "+to_string(i)+" out of range of [0,"+to_string(n0-1)+"]");)
      CNINE_CHECK_RANGE(if(i<0 || i>=n1) 
	  throw std::out_of_range("cnine::RtensorView:slice01(int): index "+to_string(i)+" out of range of [0,"+to_string(n1-1)+"]");)
	return Rtensor2_view(arr+i*s0+j*s1,n2,n3,s2,s3,dev);
    }

    Rtensor2_view slice02(const int i, const int j){
      CNINE_CHECK_RANGE(if(i<0 || i>=n0) 
	  throw std::out_of_range("cnine::RtensorView:slice02(int): index "+to_string(i)+" out of range of [0,"+to_string(n0-1)+"]");)
      CNINE_CHECK_RANGE(if(i<0 || i>=n2) 
	  throw std::out_of_range("cnine::RtensorView:slice02(int): index "+to_string(i)+" out of range of [0,"+to_string(n2-1)+"]");)
	return Rtensor2_view(arr+i*s0+j*s2,n1,n3,s1,s3,dev);
    }

    Rtensor2_view slice03(const int i, const int j){
      CNINE_CHECK_RANGE(if(i<0 || i>=n0) 
	  throw std::out_of_range("cnine::RtensorView:slice03(int): index "+to_string(i)+" out of range of [0,"+to_string(n0-1)+"]");)
      CNINE_CHECK_RANGE(if(i<0 || i>=n3) 
	  throw std::out_of_range("cnine::RtensorView:slice03(int): index "+to_string(i)+" out of range of [0,"+to_string(n3-1)+"]");)
	return Rtensor2_view(arr+i*s0+j*s3,n1,n2,s1,s2,dev);
    }

    Rtensor2_view slice12(const int i, const int j){
      CNINE_CHECK_RANGE(if(i<0 || i>=n1) 
	  throw std::out_of_range("cnine::RtensorView:slice12(int): index "+to_string(i)+" out of range of [0,"+to_string(n1-1)+"]");)
	CNINE_CHECK_RANGE(if(i<0 || i>=n2) 
	    throw std::out_of_range("cnine::RtensorView:slice12(int): index "+to_string(i)+" out of range of [0,"+to_string(n2-1)+"]");)
	return Rtensor2_view(arr+i*s1+j*s2,n0,n3,s0,s3,dev);
    }

    Rtensor2_view slice13(const int i, const int j){
      CNINE_CHECK_RANGE(if(i<0 || i>=n1) 
	  throw std::out_of_range("cnine::RtensorView:slice13(int): index "+to_string(i)+" out of range of [0,"+to_string(n1-1)+"]");)
      CNINE_CHECK_RANGE(if(i<0 || i>=n3) 
	  throw std::out_of_range("cnine::RtensorView:slice13(int): index "+to_string(i)+" out of range of [0,"+to_string(n3-1)+"]");)
	return Rtensor2_view(arr+i*s1+j*s3,n0,n2,s0,s2,dev);
    }

    Rtensor2_view slice23(const int i, const int j){
      CNINE_CHECK_RANGE(if(i<0 || i>=n2) 
	  throw std::out_of_range("cnine::RtensorView:slice23(int): index "+to_string(i)+" out of range of [0,"+to_string(n2-1)+"]");)
      CNINE_CHECK_RANGE(if(i<0 || i>=n3) 
	  throw std::out_of_range("cnine::RtensorView:slice23(int): index "+to_string(i)+" out of range of [0,"+to_string(n3-1)+"]");)
	return Rtensor2_view(arr+i*s2+j*s3,n0,n1,s0,s1,dev);
    }
    */


  public: // ---- Conversions -------------------------------------------------------------------------------


    /*
    Gtensor<float> gtensor() const{
      Gtensor<float> R({n0,n1,n2,n3},fill::raw);
      for(int i0=0; i0<n0; i0++)
	for(int i1=0; i1<n1; i1++)
	  for(int i2=0; i2<n2; i2++)
	    for(int i3=0; i3<n3; i3++)
	      R(i0,i1,i2,i3)=(*this)(i0,i1,i2,i3);
      return R;
    }
    */
    
    
  public: // ---- I/O ----------------------------------------------------------------------------------------

  
    string str(const string indent="") const{
      return ("");
      //return gtensor().str(indent);
    }

    friend ostream& operator<<(ostream& stream, const RtensorView& x){
      stream<<x.str(); return stream;
    }


  };

  /*
  inline RtensorView split0(const Rtensor3_view& x, const int i, const int j){
    assert(i*j==x.n0);
    return RtensorView(x.arr,i,j,x.n1,x.n2,x.s0*j,x.s0,x.s1,x.s2,x.dev);
  }

  inline RtensorView split1(const Rtensor3_view& x, const int i, const int j){
    assert(i*j==x.n1);
    return RtensorView(x.arr,x.n0,i,j,x.n2,x.s0,x.s1*j,x.s1,x.s2,x.dev);
    }

  inline RtensorView split2(const Rtensor3_view& x, const int i, const int j){
    assert(i*j==x.n2);
    return RtensorView(x.arr,x.n0,x.n1,i,j,x.s0,x.s1,x.s2*j,x.s2,x.dev);
  }
  */

}


#endif 
