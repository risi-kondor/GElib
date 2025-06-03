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


#ifndef _CnineTensorSarrView
#define _CnineTensorSarrView

#include "Cnine_base.hpp"
#include "TensorView.hpp"
#include "SparseTensor.hpp"

//#include <unordered_map>
//#include "Hvector.hpp"
//#include "Gdims.hpp"
//#include "IntTensor.hpp"
//#include "RtensorA.hpp"
//#include "CSRmatrix.hpp"


namespace cnine{


  template<typename TYPE>
  class TensorSArrayView: public TensorView<TYPE>{
  public:

    typedef TensorView<TYPE> TensorView;

    MemArr<TYPE> arr;
    SparseTensor<int> offs;
    Gdims adims;
    Gdims ddims;
    GstridesB dstrides;


  public: // ---- Constructors ------------------------------------------------------------------------------


    TensorSArrayView(const MemArr<TYPE>& _arr, const SparseTensor<int>& _offs, const Gdims& _ddims, const GstridesB& _dstrides):
      arr(_arr),
      offs(_offs),
      ddims(_ddims),
      dstrides(_dstrides){}


  public: // ---- Constructors for non-view child classes ---------------------------------------------------


    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    TensorSArrayView(const Gdims& _adims, const Gdims& _ddims, const map_of_lists<int,int>& mask, const FILLTYPE& fill, const int _dev=0):
      TensorView(_adims.prepend(mask.total()),fill,_dev),
      offs(adims,mask,fill_sequential()),
      adims(_adims),
      ddims(_ddims){
      dstrides=TensorView::strides.chunk(1);
    }

    TensorSArrayView(const Gdims& _adims, const Gdims& _ddims, const SparseTensor<int>& _offs, const int _dev=0):
      TensorView(_adims.prepend(_offs.nfilled()),_dev),
      offs(_offs),
      adims(_adims),
      ddims(_ddims){
      int i=0; offs.for_each_nonzero([&](const Gindex& ix, int& x){x=i++;});
      dstrides=TensorView::strides.chunk(1);
    }

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    TensorSArrayView(const Gdims& _adims, const Gdims& _ddims, const SparseTensor<int>& _offs, const FILLTYPE& fill, const int _dev=0):
      TensorView(_adims.prepend(_offs.nfilled()),fill,_dev),
      offs(_offs),
      adims(_adims),
      ddims(_ddims){
      int i=0; offs.for_each_nonzero([&](const Gindex& ix, int& x){x=i++;});
      dstrides=TensorView::strides.chunk(1);
    }


  public: // ---- Access -------------------------------------------------------------------------------------
    

    TensorView operator()(const Gindex& ix) const{
      CNINE_ASSRT(offs.is_filled(ix));
      return TensorView(arr+offs(ix),ddims,dstrides);
    }


  public: // ---- Lambdas ------------------------------------------------------------------------------------


    void for_each_cell(const std::function<void(const Gindex&, const TensorView&)>& lambda) const{
      offs.for_each_nonzero([&](const Gindex& ix, const int dummy){
	  lambda(ix,(*this)(ix));
	});
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "TensorSArrayView";
    }

    string str(const string indent="") const{
      ostringstream oss;
      for_each_cell([&](const Gindex& ix, const TensorView& x){
	  oss<<indent<<"Cell"<<ix<<":"<<endl;
	  oss<<x.str(indent+"  ")<<endl;
	});
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const TensorSArrayView<TYPE>& x){
      stream<<x.str(); return stream;
    }

  };

}

#endif 
