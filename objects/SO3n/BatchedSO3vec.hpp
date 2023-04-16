// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibBatchedSO3vec
#define _GElibBatchedSO3vec

#include "GElib_base.hpp"
//#include "TensorView.hpp"
//#include "TensorVirtual.hpp"
#include "Gvec.hpp"
#include "BatchedGvec.hpp"
#include "BatchedSO3vecView.hpp"

// BatchedSO3vec<RTYPE> -> Gvec<BatchedSO3vecView<RTYPE> > -> BatchedSO3vecView<RTYPE> -> 
// BatchedGvecView<int,BatchedSO3partView<RTYPE>,BatchedSO3vecView<RTYPE> > -> GvecView<<int,BatchedSO3partView<RTYPE> > 

namespace GElib{

  template<typename RTYPE>
  class BatchedSO3vec:public Gvec<BatchedSO3vecView<RTYPE> >{
  public:

    typedef cnine::Gdims Gdims;
    typedef cnine::GstridesB GstridesB;

    typedef cnine::MemArr<complex<RTYPE> > MemArr;
    typedef Gvec<BatchedSO3vecView<RTYPE> > Gvec;

    using Gvec::parts;



  public: // ---- Constructors -------------------------------------------------------------------------------


    BatchedSO3vec(){}

    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    BatchedSO3vec(const int _b, const SO3type& _tau, const FILLTYPE& fill, const int _dev=0){
      for(int l=0; l<_tau.size(); l++){
	Gdims dims({2*l+1,_tau[l]});
	parts[l]=new BatchedSO3partView<RTYPE>(_b,dims,fill,_dev);
      }
    }


  public: // ---- Named constructors -------------------------------------------------------------------------


    static BatchedSO3vec zero(const int _b, const SO3type& _tau, const int _dev=0){
      return BatchedSO3vec(_b,_tau,cnine::fill_zero(),_dev);
    }

    static BatchedSO3vec sequential(const int _b, const SO3type& _tau, const int _dev=0){
      return BatchedSO3vec(_b,_tau,cnine::fill_sequential(),_dev);
    }

    static BatchedSO3vec gaussian(const int _b, const SO3type& _tau, const int _dev=0){
      return BatchedSO3vec(_b,_tau,cnine::fill_gaussian(),_dev);
    }



  };


  // ---- Functions ------------------------------------------------------------------------------------------


  template<typename RTYPE>
  inline BatchedSO3vec<RTYPE> operator+(const BatchedSO3vec<RTYPE>& x, const BatchedSO3vec<RTYPE>& y){
    BatchedSO3vec<RTYPE> r(x);
    r.add(y);
    return r;
  }

  template<typename TYPE>
  inline BatchedSO3vec<TYPE> CGproduct(const BatchedSO3vecView<TYPE>& x, const BatchedSO3vecView<TYPE>& y, const int maxl=-1){
    GELIB_ASSRT(x.getb()==y.getb());
    BatchedSO3vec<TYPE> R=BatchedSO3vec<TYPE>::zero(x.getb(),
      GElib::CGproduct(x.get_tau(),y.get_tau(),maxl),x.device());
    add_vCGproduct(R,x,y);
    return R;
  }


}

#endif 
