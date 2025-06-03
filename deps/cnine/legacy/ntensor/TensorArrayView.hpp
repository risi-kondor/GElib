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


#ifndef _CnineTensorArrayView
#define _CnineTensorArrayView

#include "Cnine_base.hpp"
#include "TensorView.hpp"

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
  class TensorArrayView: public TensorView<TYPE>{
  public:

    typedef TensorView<TYPE> _TensorView;

    //using _TensorView::_TensorView;
    using _TensorView::arr;
    using _TensorView::dims;
    using _TensorView::strides;
    using _TensorView::dev;
    
    using _TensorView::device;
    using _TensorView::total;


    //using _TensorView::unsqueeze;
    //using _TensorView::cinflate;

    int ak=0;


  public: // ---- Constructors ------------------------------------------------------------------------------


    TensorArrayView(){}

    TensorArrayView(const MemArr<TYPE>& _arr, const int _ak, const Gdims& _dims, const GstridesB& _strides):
      _TensorView(_arr,_dims,_strides), ak(_ak){}


  public: // ---- Constructors for non-view child classes ---------------------------------------------------


    TensorArrayView(const int _ak, const Gdims& _dims, const int _dev=0):
      _TensorView(_dims,_dev),ak(_ak){}

    TensorArrayView(const Gdims& _adims, const Gdims& _dims, const int _dev=0):
      _TensorView(_adims.cat(_dims),_dev),ak(_adims.size()){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    TensorArrayView(const Gdims& _adims, const Gdims& _dims, const FILLTYPE& fill, const int _dev=0):
      _TensorView(_adims.cat(_dims),fill,_dev), ak(_adims.size()){}


  public: // ---- Copying -----------------------------------------------------------------------------------


    TensorArrayView& operator=(const TensorArrayView& x) const{
      _TensorView::operator=(x);
      const_cast<TensorArrayView&>(*this).ak=x.ak;
      return const_cast<TensorArrayView&>(*this);
    }

    TensorArrayView* clone() const{
      auto r=new TensorArrayView(MemArr<TYPE>(dims.total(),dev),ak,dims,GstridesB(dims));
      (*r)=*this;
      return r;
    }


  public: // ---- Devices ------------------------------------------------------------------------------------


    TensorArrayView(const TensorArrayView<TYPE>& x, const int _dev):
      _TensorView(x,_dev), ak(x.ak){}


  public: // ---- Conversions --------------------------------------------------------------------------------


    // probably deprecated
    //TensorArrayView(const _TensorView& x, const Gdims& _adims):
    //_TensorView(x.arr,_adims.cat(x.dims),GstridesB(_adims.size(),fill_zero()).cat(x.strides)), ak(_adims.size()){
    //}

    TensorArrayView(const int _ak, const _TensorView& x):
      _TensorView(x), ak(_ak){}

    TensorArrayView(const Gdims& _adims, const _TensorView& x):
      _TensorView(x.arr,_adims.cat(x.dims),GstridesB(_adims.size(),fill_zero()).cat(x.strides)),ak(_adims.size()){
    }

    TensorArrayView(const _TensorView& x, const Gdims& _ddims):
      _TensorView(x.arr,x.dims.cat(_ddims),x.strides.cat(GstridesB(_ddims.size(),fill_zero()))),ak(x.ndims()){
    }


  public: // ---- ATen --------------------------------------------------------------------------------------


    #ifdef _WITH_ATEN

    TensorArrayView(const int _ak, const at::Tensor& T):
      TensorArrayView(_ak,_TensorView(T)){}

    #endif 

    
  public: // ---- Access -------------------------------------------------------------------------------------


    int nadims() const{
      return ak;
    }

    int nddims() const{
      return dims.size()-ak;
    }

    Gdims get_adims() const{
      return dims.chunk(0,ak);
    }

    int adim(const int i) const{
      return dims(i);
    }

    Gdims get_ddims() const{
      return dims.chunk(ak);
    }

    int ddim(const int i) const{
      return dims(ak+i);
    }

    Gdims get_astrides() const{
      return strides.chunk(0,ak);
    }

    GstridesB get_dstrides() const{
      return strides.chunk(ak);
    }

    int getN() const{
      return total()/strides[ak-1];
    }


    _TensorView operator()(const int i0){
      CNINE_ASSRT(ak==1);
      return _TensorView(arr+strides[0]*i0,get_ddims(),get_dstrides());
    }

    _TensorView operator()(const int i0, const int i1){
      CNINE_ASSRT(ak==2);
      return _TensorView(arr+strides[0]*i0+strides[1]*i1,get_ddims(),get_dstrides());
    }

    _TensorView operator()(const int i0, const int i1, const int i2){
      CNINE_ASSRT(ak==3);
      return _TensorView(arr+strides[0]*i0+strides[1]*i1+strides[2]*i2,get_ddims(),get_dstrides());
    }

    _TensorView operator()(const Gindex& ix) const{
      CNINE_ASSRT(ix.size()==ak);
      return _TensorView(arr+strides(ix),get_ddims(),get_dstrides());
    }


  public: // ---- Lambdas ------------------------------------------------------------------------------------


    void for_each_cell(const std::function<void(const Gindex&, const _TensorView& x)>& lambda) const{
      get_adims().for_each_index([&](const Gindex& ix){
	  lambda(ix,(*this)(ix));});
    }

    void for_each_cell(const TensorArrayView& x, 
      const std::function<void(const Gindex&, const _TensorView& r, const _TensorView& x)>& lambda) const{
      get_adims().for_each_index([&](const Gindex& ix){
	  lambda(ix,(*this)(ix),x(ix));});
    }

    void for_each_cell(const TensorArrayView& x, const TensorArrayView& y, 
      const std::function<void(const Gindex&, const _TensorView& r, const _TensorView& x, const _TensorView& y)>& lambda) const{
      get_adims().for_each_index([&](const Gindex& ix){
	  lambda(ix,(*this)(ix),x(ix),y(ix));});
    }


    void apply_as_mvprod(const TensorArrayView& x, const TensorArrayView& y, 
      const std::function<void(const _TensorView&, const _TensorView&, const _TensorView&)>& lambda){
      CNINE_ASSRT(nadims()==1);
      CNINE_ASSRT(x.nadims()==2);
      CNINE_ASSRT(y.nadims()==1);
      CNINE_ASSRT(x.dims[0]==dims[0]);
      CNINE_ASSRT(x.dims[1]==y.dims[0]);
      for(int i=0; i<dims[0]; i++)
	for(int j=0; j<dims[1]; j++)
	  lambda((*this)(i),x(i,j),y(j));
    }

    void apply_as_mmprod(const TensorArrayView& x, const TensorArrayView& y, 
      const std::function<void(const _TensorView&, const _TensorView&, const _TensorView&)>& lambda){
      CNINE_ASSRT(nadims()==2);
      CNINE_ASSRT(x.nadims()==2);
      CNINE_ASSRT(y.nadims()==2);
      CNINE_ASSRT(dims[0]==x.dims[0]);
      CNINE_ASSRT(x.dims[1]==y.dims[0]);
      CNINE_ASSRT(dims[1]==y.dims[1]);

      int I=dims[0];
      int J=dims[1];
      int K=x.dims[1];
      for(int i=0; i<I; i++)
	for(int j=0; j<J; j++)
	  for(int k=0; k<K; k++)
	    lambda((*this)(i,j),x(j,k),y(k,j));
    }


  public: // ---- Reshapings ---------------------------------------------------------------------------------

    
    TensorArrayView permute_adims(const vector<int>& p){
      return TensorArrayView(arr,get_adims().permute(p).cat(get_ddims()),
	get_astrides().permute(p).get_dstrides());
    }

    TensorArrayView aflatten() const{
      return TensorArrayView(arr,get_ddims().prepend(getN()),get_dstrides().prepend(strides[ak-1]));
    }

    TensorArrayView<TYPE> unsqueeze(const int d) const{
      if(d<=ak) return TensorArrayView(ak+1,_TensorView::unsqueeze(d));
      else return TensorArrayView(ak,_TensorView::unsqueeze(d));
    }

    //TensorArrayView<TYPE> insert_dim(const int d, const int n) const{
    //return TensorArrayView(ak,insert_dim(d,n));
    //}

    TensorArrayView<TYPE> cinflate(const int d, const int n) const{
      return TensorArrayView(ak,_TensorView::cinflate(d,n));
    }


  public: // ---- Cumulative Operations ----------------------------------------------------------------------


    //void add(const _TensorView& x) const{
    //add(TensorArrayView(x,get_adims()));
    //}


  public: // ---- Scalar valued operations ------------------------------------------------------------------


    TYPE diff2(const TensorArrayView& x){
      TYPE t=0;
      for_each_cell(x,[&](const int b, const auto& _x, const auto& _y){
	  t+=_x.diff2(_y);});
      return t;
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "TensorArrayView";
    }

    string describe() const{
      ostringstream oss;
      oss<<"TensorArrayView"<<dims<<" ["<<strides<<"]"<<endl;
      return oss.str();
    }

    string str(const string indent="") const{
      if(dev>0) return TensorArrayView(*this,0).str(indent);
      ostringstream oss;
      for_each_cell([&](const Gindex& ix, const _TensorView& x){
	  oss<<indent<<"Cell"<<ix<<":"<<endl;
	  oss<<x.str(indent+"  ")<<endl;
	});
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const TensorArrayView<TYPE>& x){
      stream<<x.str(); return stream;
    }

  };


}

#endif


