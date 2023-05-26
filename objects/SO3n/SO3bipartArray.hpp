// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibSO3bipartArray
#define _GElibSO3bipartArray

#include "GElib_base.hpp"
#include "diff_class.hpp"
#include "SO3bipartArrayView.hpp"
#include "TensorArrayVirtual.hpp"
#include "SO3partArrayC.hpp"
#include "SO3templates.hpp"


namespace GElib{


  // SO3bipartArray > TensorArrayVirtual > SO3bipartArrayView > BatchedTensorArrayView > 
  // TensorArrayView > TensorView 

  template<typename TYPE>
  class SO3bipartArray: public cnine::TensorArrayVirtual<complex<TYPE>, SO3bipartArrayView<TYPE> >,
		      public cnine::diff_class<SO3bipartArray<TYPE> >{
    
  public:

    //typedef cnine::device device;
    typedef cnine::fill_pattern fill_pattern;
    typedef cnine::fill_zero fill_zero;

    typedef cnine::Gdims Gdims;
    typedef cnine::TensorArrayVirtual<complex<TYPE>, SO3bipartArrayView<TYPE> > TensorArrayVirtual;
    typedef SO3bipartArrayView<TYPE> _SO3bipartArrayView;

    using TensorArrayVirtual::TensorArrayVirtual;
    using TensorArrayVirtual::arr;
    //using TensorArrayVirtual::device;
    using TensorArrayVirtual::move_to_device;

    using _SO3bipartArrayView::getl1;
    using _SO3bipartArrayView::getl2;
    using _SO3bipartArrayView::getn;
    //using _SO3bipartArrayView::dim;
    //using _SO3bipartArrayView::device;


    ~SO3bipartArray(){
    }


  public: // ---- Constructors --------------------------------------------------------------------------------


    //SO3bipartArray(const int _b, const Gdims& _dims, const int l, const int n, const int _dev=0):
    //TensorArrayVirtual(_b,_dims,{2*l1+1,2*l2+1,n},_dev){}
    

  public: // ---- Named constructors --------------------------------------------------------------------------

    
    static SO3bipartArray raw(const int _b, const Gdims& _dims, const int l1, const int l2, const int c, const int _dev=0){
      return SO3bipartArray(_b,_dims,{2*l1+1,2*l2+1,c},cnine::fill_raw(),_dev);}
    
    static SO3bipartArray zero(const int _b, const Gdims& _dims, const int l1, const int l2, const int c, const int _dev=0){
      return SO3bipartArray(_b,_dims,{2*l1+1,2*l2+1,c},cnine::fill_zero(),_dev);}
    
    static SO3bipartArray sequential(const int _b, const Gdims& _dims, const int l1, const int l2, const int c, const int _dev=0){
      return SO3bipartArray(_b,_dims,{2*l1+1,2*l2+1,c},cnine::fill_sequential(),_dev);}
    
    static SO3bipartArray gaussian(const int _b, const Gdims& _dims, const int l1, const int l2, const int c, const int _dev=0){
      return SO3bipartArray(_b,_dims,{2*l1+1,2*l2+1,c},cnine::fill_gaussian(),_dev);
      //return _SO3bipartArrayView(_b,_dims,l,c,cnine::fill_gaussian(),_dev);
      //return TensorArrayVirtual(_dims.prepend(_b),Gdims({2*l1+1,2*l2+1,c}),cnine::fill_gaussian(),_dev);
    }
    

    static SO3bipartArray* new_zeros_like(const SO3bipartArray& x){
      return new SO3bipartArray(x.getb(),x.get_adims(),Gdims({2*x.getl1()+1,2*x.getl2()+1,x.getn()}),cnine::fill_zero(),x.device());
    }


  public: // ---- Conversions ---------------------------------------------------------------------------------


    //SO3bipartArray(const TensorArrayVirtual& x):
    //TensorArrayVirtual(x){}


  public: // ---- ATen --------------------------------------------------------------------------------------


    #ifdef _WITH_ATEN

    SO3bipartArray(const at::Tensor& T):
      TensorArrayVirtual(T.dim()-2,T){}

    #endif 

    
  public: // ---- Access --------------------------------------------------------------------------------------



  public: // ---- CG-products --------------------------------------------------------------------------------


  };


  template<typename TYPE>
  inline SO3partArray<TYPE> CGtransform(const SO3bipartArrayView<TYPE>& x, const int l){
    assert(l>=abs(x.getl1()-x.getl2()) && l<=x.getl1()+x.getl2());
    SO3partArray<TYPE> r=SO3partArray<TYPE>::zero(x.getb(),x.get_adims(),l,x.getn(),x.device());
    x.add_CGtransform_to(r);
    return r;
  }



}

#endif 

