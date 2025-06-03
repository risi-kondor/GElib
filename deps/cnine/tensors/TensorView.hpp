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


#ifndef _CnineTensorView
#define _CnineTensorView

#include "Cnine_base.hpp"
#include "ExprTemplates.hpp"

#include "Gdims.hpp"
#include "GstridesB.hpp"
#include "Gindex.hpp"
#include "MemArr.hpp"
#include "device_helpers.hpp"
#include "NamedTypes.hpp"

#include "Rtensor1_view.hpp"
#include "Rtensor2_view.hpp"
#include "Rtensor3_view.hpp"

#include "Ctensor1_view.hpp"
#include "Ctensor2_view.hpp"
#include "Ctensor3_view.hpp"

#include "Itensor1_view.hpp"
#include "Itensor2_view.hpp"
#include "Itensor3_view.hpp"

#include "tensor1_view.hpp"

#include "TensorView_assign.hpp"
#include "TensorView_add.hpp"

#ifdef _WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif 

#ifdef _WITH_CUBLAS
#include <cublas_v2.h>
extern cublasHandle_t cnine_cublas;
#endif 

#ifdef _WITH_include
#include <Eigen/Dense>
#endif


namespace cnine{

  extern string base_indent;

  template<typename TYPE> class TensorView;
  template<typename TYPE> class Tensor;
  template<typename TYPE> class TensorArrayView;
  template<typename TYPE> class TensorSArrayView;
  template<typename TYPE> class Tensor;
  template<typename TYPE> class Ltensor;
  template<typename TYPE> class BatchedTensorView;

#ifdef _WITH_CUDA
  template<typename TYPE>
  extern void TensorView_inc_cu(const TensorView<TYPE>& r, const TYPE v, const cudaStream_t& stream);
#endif 

  // this is the multiply defined functions problem
  inline Itensor1_view view1_of(const TensorView<int>& x);
  inline Rtensor1_view view1_of(const TensorView<float>& x);
  inline Rtensor1_view view1_of(const TensorView<double>& x); // hack
  inline Ctensor1_view view1_of(const TensorView<complex<float> >& x);

  inline Itensor2_view view2_of(const TensorView<int>& x);
  inline Rtensor2_view view2_of(const TensorView<float>& x);
  inline Rtensor2_view view2_of(const TensorView<double>& x); // hack
  inline Ctensor2_view view2_of(const TensorView<complex<float> >& x);

  inline Itensor3_view view3_of(const TensorView<int>& x);
  inline Rtensor3_view view3_of(const TensorView<float>& x);
  inline Rtensor3_view view3_of(const TensorView<double>& x); //hack
  inline Ctensor3_view view3_of(const TensorView<complex<float> >& x);

  inline Rtensor1_view flat_view_of(const TensorView<float>& x);


  template<typename TYPE>
  class TensorView{
  public:

    friend class Tensor<TYPE>;
    friend class TensorArrayView<TYPE>;
    friend class TensorSArrayView<TYPE>;
    friend class Tensor<TYPE>;
    friend class Ltensor<TYPE>;
    friend class BatchedTensorView<TYPE>;

    typedef std::size_t size_t;


    MemArr<TYPE> arr;
    Gdims dims;
    GstridesB strides;
    int dev;

  
  public: // ---- Constructors ------------------------------------------------------------------------------


    TensorView(){}

    TensorView(const MemArr<TYPE>& _arr, const Gdims& _dims, const GstridesB& _strides):
      arr(_arr),
      dims(_dims), 
      strides(_strides), 
      dev(_arr.device()){
    }

    TensorView(const Gdims& _dims, const GstridesB& _strides, const int _dev):
      arr(_strides.memsize(_dims),_dev),
      dims(_dims), 
      strides(_strides), 
      dev(_dev){
    }


  public: // ---- Constructors ------------------------------------------------------------------------------


    TensorView(const Gdims& _dims, const int fcode, const int _dev):
      dims(_dims),
      strides(GstridesB(_dims)),
      dev(_dev){
      FNTRACE();

      size_t N=dims.asize();

      if(fcode==0){
	arr=MemArr<TYPE>(N,fill_zero(),_dev);
	return;
      }

      if(fcode==1){
	arr=MemArr<TYPE>(N,_dev);
	return;
      }

      if(fcode>1){
	arr=MemArr<TYPE>(N,0);
	dev=0;

	switch(fcode){

	  case(2): //ones 
	    for(size_t i=0; i<N; i++)
	      arr[i]=1.0;
	  break;

	  case(3): //sequential 
	    for(size_t i=0; i<N; i++)
	      arr[i]=i;
	  break;

	  case(4): //gaussian
	    normal_distribution<double> distr;
	    if constexpr(is_complex<TYPE>()){
	      for(int i=0; i<N; i++) 
		arr[i]=TYPE(distr(rndGen),distr(rndGen));
	    }else{
	      for(int i=0; i<N; i++) 
		arr[i]=distr(rndGen);
	    }
	  break;

 	}

	move_to_device(_dev);
	return;
      }

      arr=MemArr<TYPE>(N,_dev);
    }

#include "TensorView_constructors.hpp"


  public: // ---- Copying -----------------------------------------------------------------------------------


    TensorView(const TensorView<TYPE>& x):
      arr(x.arr),
      dims(x.dims),
      strides(x.strides),
      dev(x.dev){
    }

    TensorView& operator=(const TensorView& x) const{
      CNINE_ASSRT(dims==x.dims);
      CNINE_ASSIGN_WARNING();
      TensorView_assign(*this,x);
      return const_cast<TensorView&>(*this);
    }        

    TensorView& set(const TensorView& x) const{
      (*this)=x;
      return const_cast<TensorView&>(*this);
    }

    void reset(const TensorView& x){
      arr=x.arr;
      dims=x.dims;
      strides=x.strides;
      dev=x.dev;
    }

    TensorView copy() const{
      TensorView R(dims,0,dev);
      R=*this;
      return R;
    }

    TensorView copy(const int _dev) const{
      TensorView R(dims,0,_dev);
      R=*this;
      return R;
    }

    TensorView copy_keeping_strides(int _dev=-1) const{
      if(_dev==-1) _dev=dev;
      TensorView R(MemArr<TYPE>(strides.memsize(dims),_dev),dims,strides);
      R=*this;
      return R;
    }
    
    TensorView* clone() const{
      return new TensorView(*this);
    }


  public: // ---- Devices ------------------------------------------------------------------------------------


    // change this so as to keep strides!!
    TensorView(const TensorView<TYPE>& x, const int _dev):
      TensorView<TYPE>(MemArr<TYPE>(x.dims.asize(),_dev),x.dims,GstridesB(x.dims)){
      (*this)=x;
    }

    void move_to_device(const int _dev) const{
      if(_dev==dev) return;
      TensorView t(*this,_dev);
      const_cast<MemArr<TYPE>&>(arr)=t.arr;
      const_cast<int&>(dev)=t.dev;
    }


  public: // ---- Conversions --------------------------------------------------------------------------------


    template<typename U=TYPE, typename = typename std::enable_if<std::is_same<U,float>::value, U>::type>
    operator Rtensor1_view() const{
      CNINE_ASSRT(ndims()==1);
      return Rtensor1_view(mem(),dims(0),strides(0),dev);
    }

    template<typename U=TYPE, typename = typename std::enable_if<std::is_same<U,float>::value, U>::type>
    operator Rtensor2_view() const{
      CNINE_ASSRT(ndims()==2);
      return Rtensor2_view(mem(),dims(0),dims(1),strides(0),strides(1),dev);
    }

    template<typename U=TYPE, typename = typename std::enable_if<std::is_same<U,float>::value, U>::type>
    operator Rtensor3_view() const{
      CNINE_ASSRT(ndims()==3);
      return Rtensor3_view(mem(),dims(0),dims(1),dims(2),strides(0),strides(1),strides(2),dev);
    }

    auto view1() const -> decltype(view1_of(*this)){
      CNINE_ASSRT(ndims()==1);
      return view1_of(*this);
    }

    auto view2() const -> decltype(view2_of(*this)){
      CNINE_ASSRT(ndims()==2);
      return view2_of(*this);
    }

    auto view3() const -> decltype(view3_of(*this)){
      CNINE_ASSRT(ndims()==3);
      return view3_of(*this);
    }

    auto flat_view() const -> decltype(view1_of(*this)){
      return flat_view_of(*this);
    }


  public: // ---- ATen --------------------------------------------------------------------------------------


  

  public: // ---- Eigen --------------------------------------------------------------------------------------


#ifdef _WITH_EIGEN

    operator Eigen::MatrixXf() const{
      CNINE_ASSRT(ndims()==2);
      int n=dims[0];
      int m=dims[1];
      Eigen::MatrixXf A(dims[0],dims[1]);
      for(int i=0; i<n; i++) 
	for(int j=0; j<m; j++) 
	  A(i,j)=(*this)(i,j);
      return A;
    }

    operator Eigen::MatrixXd() const{
      CNINE_ASSRT(ndims()==2);
      int n=dims[0];
      int m=dims[1];
      Eigen::MatrixXd A(dims[0],dims[1]);
      for(int i=0; i<n; i++) 
	for(int j=0; j<m; j++) 
	  A(i,j)=(*this)(i,j);
      return A;
    }

#endif 
  
  public: // ---- Access -------------------------------------------------------------------------------------


    int device() const{
      return dev;
    }
    
    int get_dev() const{
      return dev;
    }
    
    int get_device() const{
      return dev;
    }
    
    bool is_regular() const{
      return strides.is_regular(dims);
    }

    bool is_contiguous() const{
      return strides.is_contiguous(dims);
    }

    int ndims() const{
      return dims.size();
    }

    int dim(const int i) const{
      return dims(i);
    }

    int get_dim(const int i) const{
      return dims(i);
    }

    Gdims get_dims() const{
      return dims;
    }

    int stride(const int i) const{
      return strides[i];
    }

    GstridesB get_strides() const{
      return strides;
    }

    size_t asize() const{
      return dims.asize();
    }

    //size_t total() const{
    //return dims.total();
    //}

    size_t memsize() const{
      return strides.memsize(dims);
    }

    TYPE* get_arr() const{
      return const_cast<TYPE*>(arr.get_arr());
    } 

    TYPE* mem() const{
      return const_cast<TYPE*>(arr.get_arr());
    }

    template<typename TYPE2>
    TYPE2* mem_as() const{
      return const_cast<TYPE2*>(arr.template ptr_as<TYPE2>());
    }

    //TYPE& mem(const int i) const{
    //return *(const_cast<TYPE*>(arr.get_arr())+strides.offset);
    //}

    bool operator==(const TensorView<TYPE>& x) const{
      if(get_dev()!=(0)){
	auto a=TensorView<TYPE>(*this,0);
	return a==x;
      }
      if(x.get_dev()!=0){
	auto a=TensorView<TYPE>(x,0);
	return (*this)==a;
      }
      if(dims!=x.dims) return false;
      if(is_regular() && x.is_regular()){
	int N=asize();
	for(int i=0; i<N; i++)
	  if(arr[i]!=x.arr[i]) return false;
      }else{
	CNINE_UNIMPL();
	//for_each([&x](const Gindex& ix, const TYPE v){
	//  if(x(ix)!=v) return false;});
      }
      return true;
    }

    bool operator!=(const TensorView<TYPE>& x) const{
      return !((*this)==x);
    }

#include "TensorView_access.inc"
#include "TensorView_lambdas.inc"
#include "TensorView_reshapes.inc"
#include "TensorView_subtensors.inc"
#include "TensorView_scalar_ops.inc"
#include "TensorView_cumulative_ops.inc"
#include "TensorView_gen_ops.inc"
#include "TensorView_ATen.inc"
#include "TensorView_IO.inc"

  
  public: // ---- In-place Operations ------------------------------------------------------------------------


    void set_zero() const{
      if(dev==0){
	if(is_contiguous())
	  std::fill(mem(),mem()+asize(),0);
	else
	  CNINE_UNIMPL();
      }
      if(dev==1){
	if(is_contiguous()){
	  CUDA_SAFE(cudaMemset(mem(),0,asize()*sizeof(TYPE)));
	}else
	  CNINE_UNIMPL();
      }
    }


    void inplace_times(const TYPE c){
      if(dev==0){
	if(is_contiguous())
	  for(int i=0; i<asize(); i++) arr[i]*=c;
	else
	  for_each([&](const Gindex& ix, TYPE& x){x*=c;});
      }
      if(dev==1){
	if(is_contiguous()){
	  const float cr=c;
	  CUBLAS_SAFE(cublasSaxpy(cnine_cublas,asize(),&cr,mem(), 1,mem(), 1));
	}else
	  CNINE_UNIMPL();
      }
    }

    void normalize(){
      inplace_times(1.0/norm());
    }


  public: // ---- Cumulative Operations ----------------------------------------------------------------------


  

  public: // ---- Matrix multiplication ---------------------------------------------------------------------


    void add_mvprod(const TensorView& x, const TensorView& y) const{
      reconcile_devices<TensorView<TYPE> >(*this,x,y,[](const TensorView<TYPE>& r, const TensorView<TYPE>& x, const TensorView<TYPE>& y){
	  CNINE_NDIMS_IS_1(r);
	  CNINE_NDIMS_IS_2(x);
	  CNINE_NDIMS_IS_1(y);
	  CNINE_ASSRT(x.dims[0]==r.dims[0]);
	  CNINE_ASSRT(x.dims[1]==y.dims[0]);

	  if(r.dev==0){
	    for(int i=0; i<r.dims[0]; i++){
	      TYPE t=0;
	      for(int k=0; k<x.dims[1]; k++)
		t+=x(i,k)*y(k);
	      r.inc(i,t);
	    }
	  }
	  if(r.dev==1){
	    CNINE_UNIMPL();
	  }
	});
    }


    void add_mvprod_T(const TensorView& x, const TensorView& y){
      reconcile_devices<TensorView<TYPE> >(*this,x,y,[](TensorView<TYPE>& r, const TensorView<TYPE>& x, const TensorView<TYPE>& y){
	  CNINE_NDIMS_IS_1(r);
	  CNINE_NDIMS_IS_2(x);
	  CNINE_NDIMS_IS_1(y);
	  CNINE_ASSRT(x.dims[1]==r.dims[0]);
	  CNINE_ASSRT(y.dims[0]==y.dims[0]);

	  if(r.dev==0){
	    for(int i=0; i<r.dims[0]; i++){
	      TYPE t=0;
	      for(int k=0; k<x.dims[1]; k++)
		t+=x(k,i)*y(k);
	      r.inc(i,t);
	    }
	  }
	  if(r.dev==1){
	    CNINE_UNIMPL();
	  }
	});
    }

  
    void add_mprod(const TensorView& x, const TensorView& y) const{
      reconcile_devices<TensorView<TYPE> >(*this,x,y,[]
	(const TensorView<TYPE>& r, const TensorView<TYPE>& x, const TensorView<TYPE>& y){
	  CNINE_NDIMS_IS_2(r);
	  CNINE_NDIMS_IS_2(x);
	  CNINE_NDIMS_IS_2(y);
	  CNINE_ASSRT(x.dims[0]==r.dims[0]);
	  CNINE_ASSRT(y.dims[1]==r.dims[1]);
	  CNINE_ASSRT(x.dims[1]==y.dims[0]);

	  r.view2().add_matmul_AA(x.view2(),y.view2());

	  /*
	  if(r.dev==0){
	    for(int i=0; i<r.dims[0]; i++)
	      for(int j=0; j<r.dims[1]; j++){
		TYPE t=0;
		for(int k=0; k<x.dims[1]; k++)
		  t+=x(i,k)*y(k,j);
		r.inc(i,j,t);
	      }
	  }
	  if(r.dev==1){
	    CNINE_UNIMPL();
	  }
	  */

	});
    }


  public: // ---- Tensor multiplication ---------------------------------------------------------------------
  

    void add_tprod(const TensorView& x, const TensorView& y) const{
      reconcile_devices<TensorView<TYPE> >(*this,x,y,[&](const TensorView<TYPE>& r, const TensorView<TYPE>& x, const TensorView<TYPE>& y){
	  //CNINE_NDIMS_IS_2(r);
	  //CNINE_NDIMS_IS_2(x);
	  //CNINE_NDIMS_IS_2(y);
	  CNINE_ASSRT(x.ndims()==y.ndims());

	  if(x.ndims()==1){
	    CNINE_ASSRT(r.dims[0]==x.dims[0]*y.dims[0]);
	      for(int i=0; i<x.dims[0]; i++)
		r.block({y.dims[0]},{i*y.dims[0]}).add(y,x(i));
	    return;
	  }
	  
	  if(x.ndims()==2){
	    CNINE_ASSRT(r.dims[0]==x.dims[0]*y.dims[0]);
	    CNINE_ASSRT(r.dims[1]==x.dims[1]*y.dims[1]);
	    for(int i=0; i<x.dims[0]; i++)
	      for(int j=0; j<x.dims[1]; j++)
		r.block({y.dims[0],y.dims[1]},{i*y.dims[0],j*y.dims[1]}).add(y,x(i,j));
	    return;
	  }

	  if(x.ndims()==3){
	    CNINE_ASSRT(r.dims[0]==x.dims[0]*y.dims[0]);
	    CNINE_ASSRT(r.dims[1]==x.dims[1]*y.dims[1]);
	    CNINE_ASSRT(r.dims[2]==x.dims[2]*y.dims[2]);
	    for(int i=0; i<x.dims[0]; i++)
	      for(int j=0; j<x.dims[1]; j++)
		for(int k=0; k<x.dims[2]; k++)
		  r.block(y.dims,{i*y.dims[0],j*y.dims[1],k*y.dims[2]}).add(y,x(i,j,k));
	    return;
	  }
	    
	  CNINE_UNIMPL();
	});
    }

  
  public: // ---- Scaling -----------------------------------------------------------------------------------

    
    void add_scale_columns(const TensorView<TYPE>& x, const TensorView& y){
      CNINE_DEVICE_SAME(x);
      CNINE_DEVICE_SAME(y);
      CNINE_ASSRT(x.ndims()==2);
      CNINE_ASSRT(y.ndims()==1);
      int n=dim(0);
      int nc=dim(1);
      CNINE_ASSRT(x.dim(0)==n);
      CNINE_ASSRT(x.dim(1)==nc);
      CNINE_ASSRT(y.dim(0)==nc);
      if(dev==0){
	for(int i=0; i<n; i++)
	  for(int j=0; j<nc; j++)
	    arr[i*nc+j]+=x.arr[i*nc+j]*y.arr[j];
      }
      if(dev==1){
	CUBLAS_SAFE(cublasSdgmm(cnine_cublas,CUBLAS_SIDE_LEFT,nc,n,get_arr(),nc,y.get_arr(),1,get_arr(),nc));
	//CUBLAS_SAFE(cublasSdgmm(cnine_cublas,CUBLAS_SIDE_LEFT,nc,tail/nc,arrg,nc,y.arr,1,R.arrg,R.nc));
      }    
    }



  public: // ---- Not in-place operations -------------------------------------------------------------------


  };


  // ---- View converters ----------------------------------------------------


  inline Itensor1_view view1_of(const TensorView<int>& x){
    return Itensor1_view(x.mem(),x.dim(0),x.stride(0),x.get_dev());}
  inline Rtensor1_view view1_of(const TensorView<float>& x){
    return Rtensor1_view(x.mem(),x.dim(0),x.stride(0),x.get_dev());}
  inline Rtensor1_view view1_of(const TensorView<double>& x){// hack!!
    return Rtensor1_view(reinterpret_cast<float*>(x.mem()),x.dim(0),x.stride(0),x.get_dev());}
  inline Ctensor1_view view1_of(const TensorView<complex<float> >& x){
    return Ctensor1_view(x.mem_as<float>(),x.mem_as<float>()+1,x.dim(0),2*x.stride(0),x.get_dev());}

  inline Itensor2_view view2_of(const TensorView<int>& x){
    return Itensor2_view(x.mem(),x.dim(0),x.dim(1),x.stride(0),x.stride(1),x.get_dev());}
  inline Rtensor2_view view2_of(const TensorView<float>& x){
    return Rtensor2_view(x.mem(),x.dim(0),x.dim(1),x.stride(0),x.stride(1),x.get_dev());}
  inline Rtensor2_view view2_of(const TensorView<double>& x){ // hack!!
    return Rtensor2_view(reinterpret_cast<float*>(x.mem()),x.dim(0),x.dim(1),x.stride(0),x.stride(1),x.get_dev());}
  inline Ctensor2_view view2_of(const TensorView<complex<float> >& x){
    return Ctensor2_view(x.mem_as<float>(),x.mem_as<float>()+1,
      x.dim(0),x.dim(1),2*x.stride(0),2*x.stride(1),x.get_dev());}

  inline Itensor3_view view3_of(const TensorView<int>& x){
    return Itensor3_view(x.mem(),x.dim(0),x.dim(1),x.dim(2),x.stride(0),x.stride(1),x.stride(2),x.get_dev());}
  inline Rtensor3_view view3_of(const TensorView<float>& x){
    return Rtensor3_view(x.mem(),x.dim(0),x.dim(1),x.dim(2),x.stride(0),x.stride(1),x.stride(2),x.get_dev());}
  inline Rtensor3_view view3_of(const TensorView<double>& x){
    return Rtensor3_view(reinterpret_cast<float*>(x.mem()),x.dim(0),x.dim(1),x.dim(2),x.stride(0),x.stride(1),x.stride(2),x.get_dev());}
  inline Ctensor3_view view3_of(const TensorView<complex<float> >& x){
    return Ctensor3_view(x.mem_as<float>(),x.mem_as<float>()+1,
      x.dim(0),x.dim(1),x.dim(2),2*x.stride(0),2*x.stride(1),2*x.stride(2),x.get_dev());}

  inline Rtensor1_view flat_view_of(const TensorView<float>& x){
    CNINE_ASSRT(x.is_contiguous());
    return Rtensor1_view(x.mem(),x.asize(),1,x.get_dev());}


}


namespace std{

  template<typename TYPE>
  struct hash<cnine::TensorView<TYPE> >{
  public:
    size_t operator()(const cnine::TensorView<TYPE>& x) const{
      size_t t=hash<cnine::Gdims>()(x.dims);
      if(x.is_regular()){
	int N=x.asize();
	for(int i=0; i<N; i++)
	  t=(t^hash<TYPE>()(x.arr[i]))<<1;
      }else{
	x.for_each([&t](const cnine::Gindex& ix, const TYPE v){
	    t=(t^hash<TYPE>()(v))<<1;});
      }
      return t;
    }
  };
}


#endif


    /*
    IF_FLOAT
    TensorView& operator=(const at::Tensor& T){
      CNINE_CONVERT_FROM_ATEN_WARNING();
      CNINE_ASSRT(dims==Gdims(T));
      CNINE_ASSRT(dev==T.type().is_cuda());
      if(dev==0){
	std::copy(T.data<c10::float>(),T.data<c10::float>()+total(),
	  reinterpret_cast<c10::float*>(arr.ptr()));
      }
      if(dev==1){
	//CUDA_SAFE(cudaMemcpy(arr.ptr(),T.data<c10::c10::complex<float>>(),total()*sizeof(c10::complex<float>),cudaMemcpyDeviceToDevice));
	CUDA_SAFE(cudaMemcpy(arr.ptr(),T.data<c10::float>(),total()*sizeof(c10::float),cudaMemcpyDeviceToDevice));
      }
      return *this;
    }
    */

    /*
    IF_CFLOAT
    TensorView& operator=(const at::Tensor& T){
      CNINE_CONVERT_FROM_ATEN_WARNING();
      CNINE_ASSRT(dims==Gdims(T));
      CNINE_ASSRT(dev==T.type().is_cuda());
      if(dev==0){
	//std::copy(T.data<TYPE>(),T.data<c10::TYPE>()+total(),arr.ptr());
	std::copy(T.data<c10::complex<float>>(),T.data<c10::complex<float>>()+total(),
	  reinterpret_cast<c10::complex<float>*>(arr.ptr()));
      }
      if(dev==1){
	//CUDA_SAFE(cudaMemcpy(arr.ptr(),T.data<c10::c10::complex<float>>(),total()*sizeof(c10::complex<float>),cudaMemcpyDeviceToDevice));
	CUDA_SAFE(cudaMemcpy(arr.ptr(),T.data<c10::complex<float>>(),total()*sizeof(c10::complex<float>),cudaMemcpyDeviceToDevice));
      }
      return *this;
    }
    */
  
    /*
    IF_FLOAT
    at::Tensor torch() const{
      CNINE_CONVERT_TO_ATEN_WARNING();
      assert(dev==0);
      int k=ndims();
      vector<int64_t> v(k); 
      for(int i=0; i<k; i++) v[i]=dims[i];
      at::Tensor R(at::zeros(v,torch::CPU(at::Float))); 
      //std::copy(arr,arr+memsize,reinterpret_cast<float*>(R.data<c10::complex<float> >()));
      std::copy(arr.ptr(),arr.ptr()+dims.total(),R.data<c10::float>());
      return R;
    }
    */

    /*
    TensorView(const MemoryManager& manager, const Gdims& _dims, const int fcode, const int _dev):
      arr(manager,_dims.total(),_dev),
      dims(_dims),
      strides(GstridesB(_dims)),
      dev(_dev){
      FNTRACE();

      CNINE_ASSRT(fcode<2);
      if(fcode==0) set_zero();
    }
    */

    //void set_value(const int i0, const TYPE x) const{
    //set(i0,x);
    //}

    // void set_value(const int i0, const int i1, const TYPE x) const{
    //set(i0,i1,x);
    //}


    /*
    void add(const TensorView& x){
      CNINE_DEVICE_SAME(x);
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      assert(asize()==x.asize());
      if(dev==0){
	if(is_regular() && x.is_regular() && strides==x.strides){
	  TYPE* ptr=const_cast<MemArr<TYPE>&>(arr).get_arr();
	  TYPE* xptr=const_cast<MemArr<TYPE>&>(x.arr).get_arr();
	  for(size_t i=0; i<asize(); i++) ptr[i]+=xptr[i];
	}else
	  for_each([&](const Gindex& ix, TYPE& v){v+=x(ix);});
      }
      if(dev==1){
	if(is_regular() && x.is_regular() && strides==x.strides){
	  const float alpha=1.0; // todo
	  if constexpr(std::is_same<TYPE,complex<float> >::value)
	    CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize(), &alpha, x.get_arr(), 1, get_arr(), 1));
	}else{
	  if(ndims()<=3){
	    if(ndims()==1) view1().add(x.view1());
	    if(ndims()==2) view2().add(x.view2());
	    if(ndims()==3) view3().add(x.view3());
	  }else{
	    cout<<strides<<x.strides<<endl;
	    CNINE_UNIMPL();
	  }
	}
      }
    }
    */
