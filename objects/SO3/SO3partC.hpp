
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibSO3partC
#define _GElibSO3partC

#include "GElib_base.hpp"
//#include "TensorView.hpp"
#include "TensorVirtual.hpp"
#include "SO3partView.hpp"


namespace GElib{

  template<typename TYPE>
  class SO3partC: public cnine::TensorVirtual<complex<TYPE>, SO3partView<TYPE> >{
  public:

    typedef cnine::fill_pattern fill_pattern;
    typedef cnine::Gdims Gdims;

    typedef cnine::TensorVirtual<complex<TYPE>, SO3partView<TYPE> > TensorVirtual;

    using TensorVirtual::TensorVirtual;


  public: // ---- Constructors -------------------------------------------------------------------------------


    SO3partC(const int l, const int n, const int _dev=0):
      SO3partC(Gdims({2*l+1,n}),_dev){}


  public: // ---- Named constructors -------------------------------------------------------------------------

    
    static SO3partC<TYPE> zero(const int l, const int n, const int _dev=0){
      return SO3partC<TYPE>(Gdims({2*l+1,n}),cnine::fill_zero(),_dev);}
    
    static SO3partC<TYPE> sequential(const int l, const int n, const int _dev=0){
      return SO3partC<TYPE>(Gdims({2*l+1,n}),cnine::fill_sequential(),_dev);}
    
    static SO3partC<TYPE> gaussian(const int l, const int n, const int _dev=0){
      return SO3partC<TYPE>(Gdims({2*l+1,n}),cnine::fill_gaussian(),_dev);}
    

  };

}


#endif 
