// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef _SnIsotypicSpace
#define _SnIsotypicSpace

#include "Tensor.hpp"
#include "cachedf.hpp"
//#include "SnBasis.hpp"

#include "IntegerPartition.hpp"

namespace GElib{

  template<typename TYPE>
  class SnIsotypicSpace: public cnine::Tensor<TYPE>{
  public:

    typedef cnine::Tensor<TYPE> _Tensor;

    using _Tensor::dims;
    using _Tensor::block;
    using _Tensor::fuse01;
    //using _Tensor::repr;
    using _Tensor::str;


    Snob2::IntegerPartition ix;


  public: // ---- Constructors ------------------------------------------------------------------------------


    SnIsotypicSpace(){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SnIsotypicSpace(const Snob2::IntegerPartition& _ix, const int m, const int n, const int p, const FILLTYPE& dummy):
      cnine::Tensor<TYPE>({m,n,p},dummy),ix(_ix){}

    SnIsotypicSpace(const Snob2::IntegerPartition& _ix, const cnine::TensorView<TYPE>& T):
      cnine::Tensor<TYPE>(T),ix(_ix){}

    //SnIsotypicSpace(const Snob2::IntegerPartition& _ix, const cnine::TensorView<TYPE>&& T):
    //cnine::Tensor<TYPE>(std::move(T)),ix(_ix){}


  public: // ---- Access ------------------------------------------------------------------------------------


    int multiplicity() const{
      return dims[1];
    }

    int drep() const{
      return dims[0];
    }

    int pdim() const{
      return dims[2];
    }


    int drho() const{
      return dims[0];
    }

    int dmult() const{
      return dims[1];
    }

    int demb() const{
      return dims[2];
    }


  public: // ---- Operations --------------------------------------------------------------------------------


    cnine::TensorView<TYPE> matrix(){
      return fuse01();
    }

    SnIsotypicSpace transform(const cnine::TensorView<TYPE>& T){
      return SnIsotypicSpace(ix,(fuse01()*T.transp()).split0(dims[0],dims[1]));
    }

    
  };

}

#endif 

