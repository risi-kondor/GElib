
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibSO3partArrayView
#define _GElibSO3partArrayView

#include "GElib_base.hpp"
#include "TensorArrayView.hpp"
#include "SO3partView.hpp"
#include "SO3part_view.hpp"

namespace GElib{

  template<typename RTYPE>
  class SO3partArrayView: public cnine::TensorArrayView<complex<RTYPE> >{
  public:

    typedef cnine::Gdims Gdims;
    typedef cnine::Gindex Gindex;

    typedef cnine::TensorArrayView<complex<RTYPE> > TensorArrayView;
    typedef SO3partView<RTYPE> SO3partView;
    
    using TensorArrayView::arr;
    using TensorArrayView::dims;
    using TensorArrayView::strides;
    using TensorArrayView::dev;
    using TensorArrayView::ak;

    using TensorArrayView::TensorArrayView;
    using TensorArrayView::device;
    using TensorArrayView::ndims;
    using TensorArrayView::get_adims;
    using TensorArrayView::get_ddims;
    using TensorArrayView::get_astrides;
    using TensorArrayView::get_dstrides;
    using TensorArrayView::getN;
    using TensorArrayView::slice;

    

  public: // ---- Conversions --------------------------------------------------------------------------------


    SO3partArrayView(const TensorArrayView& x):
      TensorArrayView(x){}

    operator SO3part3_view() const{
      return SO3part3_view(arr.template ptr_as<RTYPE>(),{getN(),dims(-2),dims(-1)},{2*strides(-3),2*strides(-2),2*strides(-1)},1,device());
    }


  public: // ---- Access --------------------------------------------------------------------------------------

    
    int getl() const{
      return (dims.back(1)-1)/2;
    }

    int getn() const{
      return dims.back(0);
    }


    SO3partView operator()(const int i0){
      CNINE_ASSRT(ak==1);
      return SO3partView(arr+strides[0]*i0,get_ddims(),get_dstrides());
    }

    SO3partView operator()(const int i0, const int i1){
      CNINE_ASSRT(ak==2);
      return SO3partView(arr+strides[0]*i0+strides[1]*i1,get_ddims(),get_dstrides());
    }

    SO3partView operator()(const int i0, const int i1, const int i2){
      CNINE_ASSRT(ak==3);
      return SO3partView(arr+strides[0]*i0+strides[1]*i1+strides[2]*i2,get_ddims(),get_dstrides());
    }

    SO3partView operator()(const Gindex& ix){
      CNINE_ASSRT(ix.size()==ak);
      return SO3partView(arr+strides(ix),get_ddims(),get_dstrides());
    }


  public: // ---- CG-products --------------------------------------------------------------------------------

    
    void add_CGproduct(const SO3partArrayView& x, const SO3partArrayView& y, const int _offs=0){
      SO3part_addCGproductFn()(*this,x,y,_offs);
    }

    void add_CGproduct_back0(const SO3partArrayView& g, const SO3partArrayView& y, const int _offs=0){
      SO3part_addCGproduct_back0Fn()(*this,g,y,_offs);
    }

    void add_CGproduct_back1(const SO3partArrayView& g, const SO3partArrayView& x, const int _offs=0){
      SO3part_addCGproduct_back1Fn()(*this,g,x,_offs);
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "SO3partArrayView";
    }

    string describe() const{
      ostringstream oss;
      oss<<"SO3partArrayView"<<dims<<" "<<strides<<""<<endl;
      return oss.str();
    }

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

    friend ostream& operator<<(ostream& stream, const SO3partArrayView& x){
      stream<<x.str(); return stream;
    }

  };

}


#endif 
