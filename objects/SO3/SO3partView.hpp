
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibSO3partView
#define _GElibSO3partView

#include "GElib_base.hpp"
#include "TensorView.hpp"

#include "SO3part3_view.hpp"

#include "SO3part_addCGproductFn.hpp"
#include "SO3part_addCGproduct_back0Fn.hpp"
#include "SO3part_addCGproduct_back1Fn.hpp"


namespace GElib{

  template<typename RTYPE>
  class SO3partView: public cnine::TensorView<complex<RTYPE> >{
  public:

    typedef cnine::TensorView<complex<RTYPE> > TensorView;

    using TensorView::TensorView;
    using TensorView::arr;
    using TensorView::dims;
    using TensorView::strides;

    using TensorView::device;

    

  public: // ---- Conversions --------------------------------------------------------------------------------


    SO3partView(const TensorView& x):
      TensorView(x){}

    operator SO3part3_view() const{
      return SO3part3_view(arr.template ptr_as<RTYPE>(),{1,dims[0],dims[1]},{2*strides[0]*dims[0],2*strides[0],2*strides[1]},1,device());
      //return SO3part3_view(arr.template ptr_as<RTYPE>(),{1,dims[0],dims[1]},strides.reals(),1,device());
    }


  public: // ---- Access --------------------------------------------------------------------------------------

    
    int getl() const{
      return (dims(0)-1)/2;
    }

    int getn() const{
      return dims(1);
    }


  public: // ---- CG-products --------------------------------------------------------------------------------

    
    void add_CGproduct(const SO3partView& x, const SO3partView& y, const int _offs=0){
      SO3part_addCGproductFn()(*this,x,y,_offs);
    }

    void add_CGproduct_back0(const SO3partView& g, const SO3partView& y, const int _offs=0){
      SO3part_addCGproduct_back0Fn()(*this,g,y,_offs);
    }

    void add_CGproduct_back1(const SO3partView& g, const SO3partView& x, const int _offs=0){
      SO3part_addCGproduct_back1Fn()(*this,g,x,_offs);
    }



  public: // ---- I/O ----------------------------------------------------------------------------------------


    
  };

}


#endif 
