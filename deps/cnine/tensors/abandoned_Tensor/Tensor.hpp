/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#ifndef _CnineTensor
#define _CnineTensor

#include "Cnine_base.hpp"
#include "TensorView.hpp"
//#include "TensorView_functions.hpp"

#ifdef _WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif 

#ifdef _WITH_CUBLAS
#include <cublas_v2.h>
extern cublasHandle_t cnine_cublas;
#endif 


namespace cnine{


  template<typename TYPE>
  class Tensor: public TensorView<TYPE>
#ifdef _WITH_CENGINE
	      , public Cengine::Cobject
#endif 
  {
  public:

    typedef std::size_t size_t;
    typedef TensorView<TYPE> BASE;

    using TensorView<TYPE>::TensorView;
    using TensorView<TYPE>::arr;
    using TensorView<TYPE>::dims;
    using TensorView<TYPE>::strides;
    using TensorView<TYPE>::dev;
    using TensorView<TYPE>::memsize;

    //using TensorView<TYPE>::operator=;
    using TensorView<TYPE>::ndims;
    using TensorView<TYPE>::dim;
    using TensorView<TYPE>::is_regular;
    using TensorView<TYPE>::set;
    using TensorView<TYPE>::row;
    using TensorView<TYPE>::transp;
    using TensorView<TYPE>::fuse01;
    using TensorView<TYPE>::split0;

    using TensorView<TYPE>::view1;
    using TensorView<TYPE>::view2;
    using TensorView<TYPE>::view3;

    using TensorView<TYPE>::str;
    

  public: // ---- Constructors ------------------------------------------------------------------------------


    Tensor():
      TensorView<TYPE>(MemArr<TYPE>(1),{1},{1}){}

    Tensor(const Gdims& _dims, const int _dev=0): // syntax conflict with Ltensor
      TensorView<TYPE>(MemArr<TYPE>(_dims.asize(),_dev),_dims,GstridesB(_dims)){}

    Tensor(const Gdims& _dims, const int fcode, const int _dev):
      BASE(_dims,fcode,_dev){}


  public: // ---- Dummy constructors ------------------------------------------------------------------------


    Tensor(const Gdims& _dims, const fill_zero& dummy, const int _dev=0): 
      TensorView<TYPE>(MemArr<TYPE>(_dims.asize(),dummy,_dev),_dims,GstridesB(_dims)){}

    Tensor(const Gdims& _dims, const fill_ones& dummy, const int _dev=0):
      Tensor(_dims,_dev){
      size_t N=dims.asize();
      for(int i=0; i<N; i++)
	arr[i]=1;
      move_to_device(_dev);
    }

    Tensor(const Gdims& _dims, const fill_constant<TYPE>& dummy, const int _dev=0):
      Tensor(_dims,_dev){
      size_t N=dims.asize();
      for(int i=0; i<N; i++)
	arr[i]=dummy.v;
      move_to_device(_dev);
    }

    Tensor(const Gdims& _dims, const fill_identity& dummy, const int _dev=0):
      Tensor(_dims,fill_zero()){
      CNINE_ASSRT(ndims()==2);
      CNINE_ASSRT(dim(0)==dim(1));
      int N=dim(0);
      for(int i=0; i<N; i++)
	set(i,i,1.0);
      move_to_device(_dev);
    }

    Tensor(const Gdims& _dims, const fill_random_unitary& dummy, const int _dev=0):
      Tensor(_dims,fill_zero(),0){
      CNINE_ASSRT(ndims()==2);
      CNINE_ASSRT(dim(0)==dim(1));
      int N=dim(0);
      for(int i=0; i<N; i++){
	auto v=Tensor({N},fill_gaussian(),0);
	for(int j=0; j<i; j++){
	  auto u=row(j); 
	  v.subtract(inp(u,v)*u);
	}
	row(i).add(v,TYPE(1.0)/norm(v));
      }
      move_to_device(_dev);
    }

    Tensor(const Gdims& _dims, const fill_sequential& dummy, const int _dev=0):
      Tensor(_dims,_dev){
      size_t N=dims.asize();
      for(int i=0; i<N; i++)
	arr[i]=i;
      move_to_device(_dev);
    }

    Tensor(const Gdims& _dims, const fill_gaussian& dummy, const int _dev=0):
      Tensor(_dims,_dev){
      size_t N=dims.asize();
      normal_distribution<double> distr;
      for(int i=0; i<N; i++) 
	arr[i]=distr(rndGen)*dummy.c;
      move_to_device(_dev);
    }


  public: // ---- Other constructors ------------------------------------------------------------------------


  Tensor(const initializer_list<initializer_list<TYPE> >& list, const int _dev=0){
    int n0=list.size();
    CNINE_ASSRT(n0>0);
    int n1=list.begin()->size();
    Tensor<TYPE> T(Gdims({n0,n1})); 
    int i=0;
    for(auto& p: list){
      int j=0;
      for(auto& q: p)
	T.set(i,j++,q);
      i++;
    }
    if(_dev>0) T.move_to_device(_dev);
    (*this)=T;
  }


  public: // ---- Named constructors ------------------------------------------------------------------------


    static Tensor<TYPE> zero(const Gdims& _dims, const int _dev=0){
      return Tensor<TYPE>(_dims,fill_zero(),_dev);
    }

    static Tensor<TYPE> ones(const Gdims& _dims, const int _dev=0){
      return Tensor<TYPE>(_dims,fill_constant<TYPE>(1),_dev);
    }

    static Tensor<TYPE> constant(const Gdims& _dims, const TYPE v, const int _dev=0){
      return Tensor<TYPE>(_dims,fill_constant<TYPE>(v),_dev);
    }

    static Tensor<TYPE> identity(const Gdims& _dims, const int _dev=0){
      return Tensor<TYPE>(_dims,fill_identity(),_dev);
    }

    static Tensor<TYPE> random_unitary(const Gdims& _dims, const int _dev=0){
      return Tensor<TYPE>(_dims,fill_random_unitary(),_dev);
    }

    static Tensor<TYPE> sequential(const Gdims& _dims, const int _dev=0){
      return Tensor<TYPE>(_dims,fill_sequential(),_dev);
    }

    static Tensor<TYPE> randn(const Gdims& _dims, const int _dev=0){
      return Tensor<TYPE>(_dims,fill_gaussian(),_dev);
    }

    static Tensor<TYPE> gaussian(const Gdims& _dims, const int _dev=0){
      return Tensor<TYPE>(_dims,fill_gaussian(),_dev);
    }

    static Tensor<TYPE> zeros_like(const TensorView<TYPE>& x, const int _dev=-1){
      if(_dev==-1) return Tensor<TYPE>(x.dims,fill_zero(),x.dev);
      else return Tensor<TYPE>(x.dims,fill_zero(),_dev);
    }

    static Tensor<TYPE> identity_like(const TensorView<TYPE>& x, const int _dev=-1){
      if(_dev==-1) return Tensor<TYPE>(x.dims,fill_identity(),x.dev);
      else return Tensor<TYPE>(x.dims,fill_identity(),_dev);
    }


  public: // ---- Copying -----------------------------------------------------------------------------------


    Tensor(const Tensor<TYPE>& x):
      Tensor(x.dims,x.dev){
      CNINE_COPY_WARNING();
      view()=x.view();
    }
        
    Tensor(const Tensor<TYPE>& x, const nowarn_flag& dummy):
      Tensor(x.dims,x.dev){
      view()=x.view();
    }
        
    Tensor(Tensor<TYPE>&& x):
      TensorView<TYPE>(x.arr,x.dims,x.strides){
      CNINE_MOVE_WARNING();
    }

  // TODO!!
    //Tensor(TensorView<TYPE>&& x):
    //TensorView<TYPE>(x.arr,x.dims,x.strides){
    //CNINE_MOVE_WARNING();
    //}
        
    Tensor& operator=(const Tensor& x){
      FNTRACE();
      CNINE_ASSIGN_WARNING();
      arr=MemArr<TYPE>(x.dims.asize(),x.dev);
      dims=x.dims;
      strides=GstridesB(dims);
      dev=x.dev;
      TensorView<TYPE>::operator=(x);
      return *this;
    }
    
    Tensor& operator=(Tensor&& x){
      CNINE_MOVEASSIGN_WARNING();
      dims=x.dims;
      strides=x.strides;
      dev=x.dev;
      arr=x.arr;
      return *this;
    }

    Tensor& operator=(TensorView<TYPE>&& x){
      CNINE_MOVEASSIGN_WARNING();
      dims=x.dims;
      strides=x.strides;
      dev=x.dev;
      arr=x.arr;
      return *this;
    }

    Tensor copy() const{
      Tensor R(dims,0,dev);
      R=*this;
      return R;
    }


public: // ---- Conversions ---------------------------------------------------------------------------------
  

  Tensor(const Rtensor1_view& x): // hack
    Tensor({x.n0},fill_zero(),x.dev){
    view1().add(x);
  }

    // Doesn't work 
    //operator BatchedTensorView<TYPE>() const{
    //return TensorView(*this);
    //}

    /*
  IF_FLOAT
  Tensor(const RtensorA& x):
    Tensor(x.dims,x.dev){
    TensorView<TYPE>::operator=(x);
  }
    */

  public: // ---- Transport -----------------------------------------------------------------------------------


    Tensor(const TensorView<TYPE>& x, const int _dev):
      Tensor(x.dims,_dev){
      CNINE_COPY_WARNING();
      view()=x;
    }

    void move_to_device(const int _dev) const{
      if(dev==_dev) return;
      const_cast<Tensor&>(*this)=Tensor(*this,_dev);
    }


  public: // ---- Resizing ----------------------------------------------------------------------------------


  void resize0(const int n){
    CNINE_ASSRT(ndims()>0);
    //if(n<=dim(0)) return;
    CNINE_ASSRT(is_regular());
    FNTRACE();
    if(n>=dim(0)){
      Tensor T(MemArr<TYPE>(n*strides[0],dev),dims,strides);
      T.block(dims)=*this;
      auto temp=arr;
      arr=T.arr;
      T.arr=temp;
    }
    dims[0]=n;
  }


public: // ---- ATen --------------------------------------------------------------------------------------


#ifdef _WITH_ATEN
    Tensor(const at::Tensor& T):
      Tensor(Gdims(T),T.type().is_cuda()){
      TensorView<TYPE>::operator=(T);
    }
#endif


  public: // ---- Eigen --------------------------------------------------------------------------------------


#ifdef _WITH_EIGEN

    Tensor(const Eigen::VectorXf& x):
      Tensor(Gdims(x.size())){
      int n=dims[0];
      for(int i=0; i<n; i++) 
	  set(i,x(i));
    }

    Tensor(const Eigen::VectorXd& x):
      Tensor(Gdims(x.size())){
      int n=dims[0];
      for(int i=0; i<n; i++) 
	  set(i,x(i));
    }

    Tensor(const Eigen::MatrixXf& x):
      Tensor(Gdims(x.rows(),x.cols())){
      int n=dims[0];
      int m=dims[1];
      for(int i=0; i<n; i++) 
	for(int j=0; j<m; j++) 
	  set(i,j,x(i,j));
    }

    Tensor(const Eigen::MatrixXd& x):
      Tensor(Gdims(x.rows(),x.cols())){
      int n=dims[0];
      int m=dims[1];
      for(int i=0; i<n; i++) 
	for(int j=0; j<m; j++) 
	  set(i,j,x(i,j));
    }

#endif 


  public: // ---- Views -------------------------------------------------------------------------------------


  Tensor(const TensorView<TYPE>& x):
    Tensor(x.dims,x.dev){
    CNINE_COPY_WARNING();
    view()=x;
  }

  template<typename TYPE2>
  Tensor(const TensorView<TYPE2>& x):
    TensorView<TYPE>(MemArr<TYPE>(x.memsize(),x.get_dev()),x.get_dims(),x.get_strides()){
    CNINE_CONVERT_WARNING();
    CNINE_ASSRT(dev==0);
    size_t N=memsize();
    for(int i=0; i<N; i++)
      arr[i]=x.get_arr()[i];
  }


    //Tensor(TensorView<TYPE>&& x):
    //Tensor(x.dims,x.dev){
    //CNINE_COPY_WARNING();
    //view()=x;
    //}

    TensorView<TYPE> view(){
      return TensorView<TYPE>(*this);
    }

    const TensorView<TYPE> view() const{
      return TensorView<TYPE>(*this);
    }


  public: // ---- Operations --------------------------------------------------------------------------------


    Tensor operator*(const TensorView<TYPE>& y) const{
      CNINE_ASSERT(ndims()==1||ndims()==2,"first operand of product must be a vector or a matrix");
      CNINE_ASSERT(y.ndims()==1||y.ndims()==2,"second operand of product must be a vector or a matrix");

      if(ndims()==1 && y.ndims()==2){
	Tensor R=zero({y.dims[1]},dev);
	R.add_mvprod(y.transp(),*this);
	return R;
      }

      if(ndims()==2 && y.ndims()==1){
	Tensor R=zero({dims[0]},dev);
	R.add_mvprod(*this,y);
	return R;
      }

      if(ndims()==2 && y.ndims()==2){
	Tensor R=zero({dims[0],y.dims[1]},dev);
	R.add_mprod(*this,y);
	return R;
      }

      return Tensor();
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "Tensor";
    }

    string describe() const{
      ostringstream oss;
      oss<<"Tensor"<<dims<<" ["<<strides<<"]"<<endl;
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Tensor<TYPE>& x){
      stream<<x.str(); return stream;
    }

  };


  // ---- Functions ----------------------------------------------------------------------------------------------



}


namespace std{

  template<typename TYPE>
  struct hash<cnine::Tensor<TYPE> >{
  public:
    size_t operator()(const cnine::Tensor<TYPE>& x) const{
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

