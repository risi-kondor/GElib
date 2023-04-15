
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibBatchedSO3partView
#define _GElibBatchedSO3partView

#include "GElib_base.hpp"
#include "BatchedTensorView.hpp"
#include "SO3partView.hpp"

#include "SO3part3_view.hpp"

#include "SO3part_addCGproductFn.hpp"
#include "SO3part_addCGproduct_back0Fn.hpp"
#include "SO3part_addCGproduct_back1Fn.hpp"


namespace GElib{

  template<typename RTYPE>
  class BatchedSO3partView: public cnine::BatchedTensorView<complex<RTYPE> >{
  public:

    typedef cnine::BatchedTensorView<complex<RTYPE> > BatchedTensorView;

    using BatchedTensorView::BatchedTensorView;
    using BatchedTensorView::arr;
    using BatchedTensorView::dims;
    using BatchedTensorView::strides;

    using BatchedTensorView::getb;
    using BatchedTensorView::device;
    

  public: // ---- Conversions --------------------------------------------------------------------------------


    BatchedSO3partView(const BatchedTensorView& x):
      BatchedTensorView(x){}

    operator SO3part3_view() const{
      return SO3part3_view(arr.template ptr_as<RTYPE>(),{dims[0],dims[1],dims[2]},
	{2*strides[0],2*strides[1],2*strides[2]},1,device());
    }


  public: // ---- Access --------------------------------------------------------------------------------------

    
    int getl() const{
      return (dims[1]-1)/2;
    }

    int getn() const{
      return dims[2];
    }

    SO3partView<RTYPE> batch(const int i) const{
      return SO3partView<RTYPE>(arr+strides[0]*i,dims.chunk(1),strides.chunk(1));
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string repr(const string indent="") const{
      return "<GElib::BatchedSO3part(b="+to_string(getb())+",l="+to_string(getl())+",n="+to_string(getn())+")>";
    }
    
    friend ostream& operator<<(ostream& stream, const BatchedSO3partView& x){
      stream<<x.str(); return stream;
    }
    
  };

}


#endif 


    //public: // ---- CG-products --------------------------------------------------------------------------------

    
    //void add_CGproduct(const BatchedSO3partView& x, const BatchedSO3partView& y, const int _offs=0){
    //SO3part_addCGproductFn()(*this,x,y,_offs);
    //}

    //void add_CGproduct_back0(const BatchedSO3partView& g, const BatchedSO3partView& y, const int _offs=0){
    //SO3part_addCGproduct_back0Fn()(*this,g,y,_offs);
    //}

    //void add_CGproduct_back1(const BatchedSO3partView& g, const BatchedSO3partView& x, const int _offs=0){
    //SO3part_addCGproduct_back1Fn()(*this,g,x,_offs);
    // }



