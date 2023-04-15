
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibBatchedSO3partArrayView
#define _GElibBatchedSO3partArrayView

#include "GElib_base.hpp"
#include "BatchedTensorArrayView.hpp"
#include "BatchedSO3partView.hpp"
#include "SO3part_view.hpp"


namespace GElib{

  template<typename RTYPE>
  class BatchedSO3partArrayView: public cnine::BatchedTensorArrayView<complex<RTYPE> >{
  public:

    typedef cnine::Gdims Gdims;
    typedef cnine::Gindex Gindex;

    typedef cnine::BatchedTensorArrayView<complex<RTYPE> > BatchedTensorArrayView;
    typedef BatchedSO3partView<RTYPE> BatchedSO3partView;
    
    using BatchedTensorArrayView::arr;
    using BatchedTensorArrayView::dims;
    using BatchedTensorArrayView::strides;
    using BatchedTensorArrayView::dev;
    using BatchedTensorArrayView::ak;

    using BatchedTensorArrayView::BatchedTensorArrayView;
    using BatchedTensorArrayView::device;
    using BatchedTensorArrayView::ndims;
    using BatchedTensorArrayView::nadims;
    using BatchedTensorArrayView::nddims;
    using BatchedTensorArrayView::get_adims;
    using BatchedTensorArrayView::get_ddims;
    using BatchedTensorArrayView::getb;
    using BatchedTensorArrayView::get_astrides;
    using BatchedTensorArrayView::get_dstrides;
    using BatchedTensorArrayView::getN;
    using BatchedTensorArrayView::slice;
    

  public: // ---- Conversions --------------------------------------------------------------------------------


    BatchedSO3partArrayView(const BatchedTensorArrayView& x):
      BatchedTensorArrayView(x){}

    operator SO3part3_view() const{
      return SO3part3_view(arr.template ptr_as<RTYPE>(),{getb()*getN(),dims(-2),dims(-1)},
	{2*strides(-3),2*strides(-2),2*strides(-1)},1,device());
    }


  public: // ---- Access --------------------------------------------------------------------------------------

    
    int getl() const{
      return (dims.back(1)-1)/2;
    }

    int getn() const{
      return dims.back(0);
    }


    SO3partArrayView<RTYPE> batch(const int i) const{
      return SO3partArrayView<RTYPE>(arr+strides[0]*i,nadims(),dims.chunk(1),strides.chunk(1));
    }


    BatchedSO3partView operator()(const int i0){
      CNINE_ASSRT(ak==2);
      return BatchedSO3partView(arr+strides[1]*i0,get_ddims(),get_dstrides());
    }

    BatchedSO3partView operator()(const int i0, const int i1){
      CNINE_ASSRT(ak==3);
      return BatchedSO3partView(arr+strides[1]*i0+strides[2]*i1,get_ddims(),get_dstrides());
    }

    BatchedSO3partView operator()(const int i0, const int i1, const int i2){
      CNINE_ASSRT(ak==4);
      return BatchedSO3partView(arr+strides[1]*i0+strides[2]*i1+strides[3]*i2,get_ddims(),get_dstrides());
    }

    BatchedSO3partView operator()(const Gindex& ix){
      CNINE_ASSRT(ix.size()==ak);
      return BatchedSO3partView(arr+strides.chunk(1)(ix),get_ddims(),get_dstrides());
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "BatchedSO3partArrayView";
    }

    string repr(const string indent="") const{
      return "<GElib::SO3part(b="+to_string(getb())+",adims="+get_adims().str()+",l="+to_string(getl())+",n="+to_string(getn())+")>";
    }
    
    friend ostream& operator<<(ostream& stream, const BatchedSO3partArrayView& x){
      stream<<x.str(); return stream;
    }

  };

}


#endif 


    //string describe() const{
    //ostringstream oss;
    //oss<<"BatchedSO3partArrayView"<<dims<<" "<<strides<<""<<endl;
    //return oss.str();
    //}

    /*
    string str(const string indent="") const{
      CNINE_CPUONLY();
      GELIB_ASSRT(ndims()>2);
      ostringstream oss;

      Gdims adims=dims.chunk(0,ndims()-2);
      adims.for_each_index([&](const Gindex& ix){
	  oss<<indent<<"Cell"<<ix<<":"<<endl;
	  oss<<slice(ix).str(indent+"  ")<<endl;
	});

      return oss.str();
    }
    */
    //public: // ---- CG-products --------------------------------------------------------------------------------

    
    //void add_CGproduct(const BatchedSO3partArrayView& x, const BatchedSO3partArrayView& y, const int _offs=0){
    //SO3part_addCGproductFn()(*this,x,y,_offs);
    //}

    //void add_CGproduct_back0(const BatchedSO3partArrayView& g, const BatchedSO3partArrayView& y, const int _offs=0){
    //SO3part_addCGproduct_back0Fn()(*this,g,y,_offs);
    //}

    //void add_CGproduct_back1(const BatchedSO3partArrayView& g, const BatchedSO3partArrayView& x, const int _offs=0){
    //SO3part_addCGproduct_back1Fn()(*this,g,x,_offs);
    //}


