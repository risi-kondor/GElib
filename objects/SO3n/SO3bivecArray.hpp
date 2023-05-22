// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibSO3bivecArray
#define _GElibSO3bivecArray

#include "GElib_base.hpp"
//#include "TensorView.hpp"
#include "TensorVirtual.hpp"
#include "GvecArray.hpp"
#include "SO3bivecArrayView.hpp"
#include "SO3vecArrayC.hpp"
#include "diff_class.hpp"


namespace GElib{

  template<typename RTYPE>
  class SO3bivecArray: public GvecArray<SO3bivecArrayView<RTYPE> >,
		       public cnine::diff_class<SO3bivecArray<RTYPE> >{
  public:

    typedef cnine::Gdims Gdims;
    typedef cnine::GstridesB GstridesB;

    typedef cnine::MemArr<complex<RTYPE> > MemArr;
    typedef GvecArray<SO3bivecArrayView<RTYPE> > _GvecArray;
    typedef SO3bivecArrayView<RTYPE> _SO3bivecArrayView;

    using _GvecArray::parts;

#ifdef _WITH_ATEN
    using _SO3bivecArrayView::torch;
#endif 


  public: // ---- Constructors -------------------------------------------------------------------------------


    SO3bivecArray(){}

    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3bivecArray(const int _b, const Gdims& _adims, const SO3bitype& _tau, const FILLTYPE& fill, const int _dev=0){
      for(auto p:_tau){
	int l1=p.first.first;
	int l2=p.first.second;
	Gdims dims({2*l1+1,2*l2+1,p.second});
	parts[p.first]=new SO3bipartArrayView<RTYPE>(_b,_adims,dims,fill,_dev);
      }
    }


  public: // ---- Named constructors -------------------------------------------------------------------------


    static SO3bivecArray zero(const int b, const Gdims& _adims, const SO3bitype& _tau, const int _dev=0){
      return SO3bivecArray(b,_adims,_tau,cnine::fill_zero(),_dev);
    }

    static SO3bivecArray sequential(const int b, const Gdims& _adims, const SO3bitype& _tau, const int _dev=0){
      return SO3bivecArray(b,_adims,_tau,cnine::fill_sequential(),_dev);
    }

    static SO3bivecArray gaussian(const int b, const Gdims& _adims, const SO3bitype& _tau, const int _dev=0){
      return SO3bivecArray(b,_adims,_tau,cnine::fill_gaussian(),_dev);
    }


    static SO3bivecArray* new_zeros_like(const SO3bivecArray& x){
      return new SO3bivecArray(x.getb(),x.get_adims(),x.get_tau(),cnine::fill_zero(),x.device());
    }


  public: // ---- ATen --------------------------------------------------------------------------------------

    
    #ifdef _WITH_ATEN

    //SO3bivecArray(const vector<at::Tensor>& v){
    //for(auto& p:v)
    //parts[(p.size(p.dim()-2)-1)/2]=new SO3partArrayView<RTYPE>(p.dim()-2,p);
    //}

    #endif 



  };


  // ---- Functions ------------------------------------------------------------------------------------------


  template<typename RTYPE>
  inline SO3bivecArray<RTYPE> operator+(const SO3bivecArray<RTYPE>& x, const SO3bivecArray<RTYPE>& y){
    SO3bivecArray<RTYPE> r(x);
    r.add(y);
    return r;
  }

  template<typename TYPE>
  inline SO3vecArray<TYPE> CGtransform(const SO3bivecArrayView<TYPE>& x, const int maxl=-1){
    SO3vecArray<TYPE> R=SO3vecArray<TYPE>::zero(x.getb(),x.get_adims(),
      GElib::CGtransform(x.get_tau(),maxl),x.device());
    x.add_CGtransform_to(R);
    return R;
  }

}

#endif 

