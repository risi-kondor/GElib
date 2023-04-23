// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibSO3partArray
#define _GElibSO3partArray

#include "GElib_base.hpp"
#include "SO3partArrayC.hpp"
#include "SO3partArrayView.hpp"
//#include "TensorPack.hpp"
#include "TensorArrayVirtual.hpp"
#include "SO3templates.hpp"


namespace GElib{


  template<typename TYPE>
  class SO3partArray: public cnine::TensorArrayVirtual<complex<TYPE>, SO3partArrayView<TYPE> >{
  public:

    typedef cnine::device device;
    typedef cnine::fill_pattern fill_pattern;
    typedef cnine::fill_zero fill_zero;

    typedef cnine::Gdims Gdims;
    typedef cnine::TensorArrayVirtual<complex<TYPE>, SO3partArrayView<TYPE> > TensorArrayVirtual;

    using TensorArrayVirtual::TensorArrayVirtual;
    using TensorArrayVirtual::arr;
    using TensorArrayVirtual::move_to_device;

    //using cnine::TensorVirtual<TYPE, SO3partArrayView<TYPE> >::TensorVirtual; 
    //using cnine::TensorVirtual<TYPE, SO3partArrayView<TYPE> >::arr; 
    //using cnine::TensorVirtual<TYPE, SO3partArrayView<TYPE> >::move_to_device;


    ~SO3partArray(){
    }


  public: // ---- Constructors --------------------------------------------------------------------------------


    SO3partArray(const int _b, const Gdims& _dims, const int l, const int n, const int _dev=0):
      TensorArrayVirtual(_b,_dims,{2*l+1,n},_dev){}


  public: // ---- Named constructors --------------------------------------------------------------------------

    
    static SO3partArray zero(const int _b, const Gdims& _dims, const int l, const int c, const int _dev=0){
      return SO3partArray(_b,_dims,{2*l+1,c},cnine::fill_zero(),_dev);}
    
    static SO3partArray sequential(const int _b, const Gdims& _dims, const int l, const int c, const int _dev=0){
      return SO3partArray(_b,_dims,{2*l+1,c},cnine::fill_sequential(),_dev);}
    
    static SO3partArray gaussian(const int _b, const Gdims& _dims, const int l, const int c, const int _dev=0){
      return SO3partArray(_b,_dims,{2*l+1,c},cnine::fill_gaussian(),_dev);}
    

  public: // ---- Conversions ---------------------------------------------------------------------------------


    //SO3partArray(const TensorArrayVirtual& x):
    //TensorArrayVirtual(x){}


  public: // ---- Access --------------------------------------------------------------------------------------



  public: // ---- CG-products --------------------------------------------------------------------------------


  };


  template<typename TYPE>
  inline SO3partArray<TYPE> CGproduct(const SO3partArrayView<TYPE>& x, const SO3partArrayView<TYPE>& y, const int l){
    assert(x.get_adims()==y.get_adims());
    assert(l>=abs(x.getl()-y.getl()) && l<=x.getl()+y.getl());
    SO3partArray<TYPE> R=SO3partArray<TYPE>::zero(x.getb(),x.get_adims(),l,x.getn()*y.getn(),x.device());
    add_CGproduct(R,x,y);
    return R;
  }

  template<typename TYPE>
  inline SO3partArray<TYPE> DiagCGproduct(const SO3partArrayView<TYPE>& x, const SO3partArrayView<TYPE>& y, const int l){
    assert(x.getn()==y.getn());
    assert(x.get_adims()==y.get_adims());
    assert(l>=abs(x.getl()-y.getl()) && l<=x.getl()+y.getl());
    SO3partArray<TYPE> R=SO3partArray<TYPE>::zero(x.getb(),x.get_adims(),l,x.getn(),x.device());
    add_DiagCGproduct(R,x,y);
    return R;
  }

}

#endif 


    /*
    void add_CGproduct(const SO3partArrayView& x, const SO3partArrayView& y, const int _offs=0){
      SO3part_addCGproductFn()(*this,x,y,_offs);
    }

    void add_CGproduct_back0(const SO3partArrayView& g, const SO3partArrayView& y, const int _offs=0){
      SO3part_addCGproduct_back0Fn()(*this,g,y,_offs);
    }

    void add_CGproduct_back1(const SO3partArrayView& g, const SO3partArrayView& x, const int _offs=0){
      SO3part_addCGproduct_back1Fn()(*this,g,x,_offs);
    }
    */
