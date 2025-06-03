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


#ifndef _CnineBatchedTensorArrayView
#define _CnineBatchedTensorArrayView

#include "Cnine_base.hpp"
#include "TensorArrayView.hpp"
#include "BatchedTensorView.hpp"

#include "Btensor_add_prodFn.hpp"
#include "Btensor_add_RprodFn.hpp"

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
  class BatchedTensorArrayView: public TensorArrayView<TYPE>{
  public:

    typedef TensorArrayView<TYPE> _TensorArrayView;
    typedef BatchedTensorView<TYPE> _BatchedTensorView;

    using _TensorArrayView::_TensorArrayView;
    //using _TensorArrayView::operator=;
    using _TensorArrayView::arr;
    using _TensorArrayView::dims;
    using _TensorArrayView::strides;
    using _TensorArrayView::dev;
    using _TensorArrayView::ak;
    
    using _TensorArrayView::device;
    using _TensorArrayView::total;
    using _TensorArrayView::slice;
    using _TensorArrayView::unsqueeze;
    using _TensorArrayView::cinflate;

    //int ak=0;


  public: // ---- Constructors ------------------------------------------------------------------------------


    BatchedTensorArrayView(){}

    //BatchedTensorArrayView(const MemArr<TYPE>& _arr, const int _b, const Gdims& _dims, const GstridesB& _strides):
    //BatchedTensorView(_arr,_dims,_strides), ak(_ak){}


  public: // ---- Constructors for non-view child classes ---------------------------------------------------


    BatchedTensorArrayView(const int _b, const Gdims& _adims, const Gdims& _dims, const int _dev=0):
      _TensorArrayView(_adims.prepend(_b),_dims,_dev){
      //ak=_adims.size()+1;
    }

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    BatchedTensorArrayView(const int _b, const Gdims& _adims, const Gdims& _dims, const FILLTYPE& fill, const int _dev=0):
      _TensorArrayView(_adims.prepend(_b),_dims,fill,_dev){
    }


  public: // ---- Copying -----------------------------------------------------------------------------------


    BatchedTensorArrayView* clone() const{
      auto r=new BatchedTensorArrayView(MemArr<TYPE>(dims.total(),dev),ak,dims,GstridesB(dims));
      (*r)=*this;
      return r;
    }


  public: // ---- Conversions --------------------------------------------------------------------------------


    BatchedTensorArrayView(const _TensorArrayView& x):
      _TensorArrayView(x){}

    BatchedTensorArrayView(const Gdims& _adims, const _BatchedTensorView& x):
      _TensorArrayView(x.arr,_adims.size()+1,_adims.prepend(x.getb()).cat(x.dims.chunk(1)),
	GstridesB(_adims.size(),fill_zero()).cat(x.strides.chunk(1)).prepend(x.strides(0))){
      //ak=_adims.size()+1;
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int getb() const{
      return dims(0);
    }

    Gdims get_bstride() const{
      return strides(0);
    }


    int nadims() const{
      return ak-1;
    }

    Gdims get_adims() const{
      return dims.chunk(1,ak-1);
    }

    int adim(const int i) const{
      return dims[i+1];
    }

    GstridesB get_astrides() const{
      return strides.chunk(1,ak-1);
    }

    int astride(const int i) const{
      return strides[i+1];
    }

    int getN() const{
      return get_adims().total();
    }


    int nddims() const{
      return dims.size()-ak;
    }

    Gdims get_ddims() const{
      return dims.chunk(ak);
    }

    int ddim(const int i) const{
      return dims[ak+i];
    }
    
    GstridesB get_dstrides() const{
      return strides.chunk(ak);
    }

    int dstride(const int i) const{
      return strides[ak+i];
    }


    _TensorArrayView batch(const int i) const{
      CNINE_CHECK_RANGE(dims.check_in_range_d(0,i,string(__PRETTY_FUNCTION__)));
      return _TensorArrayView(arr+strides[0]*i,nadims(),dims.chunk(1),strides.chunk(1));
    }


    _BatchedTensorView operator()(const int i0){
      CNINE_ASSRT(ak==1);
      return _BatchedTensorView(arr+astride(0)*i0,
	get_ddims().prepend(getb()),
	get_dstrides().prepend(get_bstride()));
    }

    _BatchedTensorView operator()(const int i0, const int i1){
      CNINE_ASSRT(ak==2);
      return _BatchedTensorView(arr+astride(0)*i0+astride(1)*i1,
	get_ddims().prepend(getb()),
	get_dstrides().prepend(get_bstride()));
    }

    _BatchedTensorView operator()(const int i0, const int i1, const int i2){
      CNINE_ASSRT(ak==3);
      return _BatchedTensorView(arr+astride(0)*i0+astride(1)*i1+astride(2)*i2,
	get_ddims().prepend(getb()),
	get_dstrides().prepend(get_bstride()));
    }

    _BatchedTensorView operator()(const Gindex& ix){
      CNINE_ASSRT(ix.size()==ak);
      return _BatchedTensorView(arr+strides(ix),get_ddims().prepend(getb()),get_dstrides().prepend(get_bstride()));
    }


  public: // ---- Lambdas ------------------------------------------------------------------------------------


    void for_each_batch(const std::function<void(const int, const _TensorArrayView&)>& lambda) const{
      int B=getb();
      for(int b=0; b<B; b++)
	lambda(b,batch(b));
    }

    void for_each_cell(const std::function<void(const Gindex&, const _BatchedTensorView&)>& lambda) const{
      get_adims().for_each_index([&](const Gindex& ix){
	  lambda(ix,(*this)(ix));});
    }


  public: // ---- Lambdas ------------------------------------------------------------------------------------


    void apply_as_mvprod(const BatchedTensorArrayView& x, const BatchedTensorArrayView& y, 
      const std::function<void(const _BatchedTensorView&, const _BatchedTensorView&, const _BatchedTensorView&)>& lambda){
      CNINE_ASSRT(nadims()==1);
      CNINE_ASSRT(x.nadims()==2);
      CNINE_ASSRT(y.nadims()==1);
      CNINE_ASSRT(x.adim(0)==adim(0));
      CNINE_ASSRT(x.adim(1)==y.adim(0));
      for(int i=0; i<adim(0); i++)
	for(int j=0; j<adim(1); j++)
	  lambda((*this)(i),x(i,j),y(j));
    }

    void apply_as_mmprod(const BatchedTensorArrayView& x, const BatchedTensorArrayView& y, 
      const std::function<void(const _BatchedTensorView&, const _BatchedTensorView&, const _BatchedTensorView&)>& lambda){
      CNINE_ASSRT(nadims()==2);
      CNINE_ASSRT(x.nadims()==2);
      CNINE_ASSRT(y.nadims()==2);
      CNINE_ASSRT(adim(0)==x.adim(0));
      CNINE_ASSRT(x.adim(1)==y.adim(0));
      CNINE_ASSRT(adim(1)==y.adim(1));

      int I=adim(0);
      int J=adim(1);
      int K=x.adim(1);
      for(int i=0; i<I; i++)
	for(int j=0; j<J; j++)
	  for(int k=0; k<K; k++)
	    lambda((*this)(i,j),x(j,k),y(k,j));
    }

  public: // ---- Reshapings ---------------------------------------------------------------------------------


   BatchedTensorArrayView aflatten() const{
     return _TensorArrayView(arr,2,get_ddims().prepend(getN()).prepend(getb()),
       get_dstrides().prepend(strides[ak-1]).prepend(strides[0]));
    }

   BatchedTensorArrayView baflatten() const{
     return _TensorArrayView(arr,1,get_ddims().prepend(getb()*getN()),get_dstrides().prepend(strides[ak-1]));
    }

   BatchedTensorArrayView permute_indices(const vector<int>& v) const{
     return _TensorArrayView(arr,ak,dims.permute(v),strides.permute(v));
   }

   BatchedTensorArrayView swap_batch_array() const{
     return aflatten().permute_indices({0,1});
   }


  public: // ---- Cumulative Operations ----------------------------------------------------------------------


    //void add(const _BatchedTensorView& x) const{
    //add(BatchedTensorArrayView(get_adims(),x));
    //}

    //void add_prod(const _BatchedTensorView& x, const _BatchedTensorView& y) const{
    //reconcile_batched_array<_BatchedTensorView>(*this,x,y,
    //[&](const auto& r, const auto& x, const auto& y){Btensor_add_prodFn()(r,x,y);},
    //[&](const auto& r, const auto& x, const auto& y){Btensor_add_RprodFn()(r,x,y);});
    //}


  public: // ---- Scalar valued operations ------------------------------------------------------------------


    TYPE diff2(const BatchedTensorArrayView& x){
      TYPE t=0;
      for_each_batch(x,[&](const int b, const auto& _x, const auto& _y){
	  t+=_x.diff2(_y);});
      return t;
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "BatchedTensorArrayView";
    }

    string describe() const{
      ostringstream oss;
      oss<<"BatchedTensorArrayView"<<dims<<" ["<<strides<<"]"<<endl;
      return oss.str();
    }

    string str(const string indent="") const{
      if(dev>0) return BatchedTensorArrayView(*this,0).str(indent);
      ostringstream oss;
      if(getb()>1){
	for_each_batch([&](const int b, const _TensorArrayView& x){
	    oss<<indent<<"Batch "<<b<<":"<<endl;
	    oss<<x.str(indent+"  ");
	  });
      }else{
	oss<<slice(0,0).str(indent);
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const BatchedTensorArrayView<TYPE>& x){
      stream<<x.str(); return stream;
    }

  };


}

#endif


