// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef _SO3group
#define _SO3group

#include <cnine/tensors>
#include <cnine/containers>

#include "GtypeD.hpp"
#include "GvecSpec.hpp"


namespace GElib{

  class SO3group;

  template<typename TYPE>
  class SO3partD;

  typedef GtypeD<SO3group> SO3typeD;


  class SO3group{
  public:

    typedef int IrrepIx;
    typedef GvecSpec<SO3group> vecSpec;
    typedef GtypeD<SO3group> TAU;


  public: // ---- General ------------------------------------------------------------------------------------


    static string Gname(){
      return "SO3";
    }

    template<typename TYPE>
    static SO3partD<TYPE> dummy_part(){
      return SO3partD<TYPE>();}


  public: // ---- Irreps -------------------------------------------------------------------------------------


    static int dim_of_irrep(const int l){
      return 2*l+1;
    }


  public: // ---- CG-products --------------------------------------------------------------------------------


    static int CGmultiplicity(const int l1, const int l2, const int l){
      return (l>=abs(l1-l2))&&(l<=l1+l2);
    }

    static void for_each_CGcomponent(const int l1, const int l2, 
      const std::function<void(const int&, const int)>& lambda){
      for(int l=std::abs(l1-l2); l<=l1+l2; l++)
	lambda(l,1);
    }

    static int CG_sign_rule(const int l1, const int l2, const int l, const int i){
      return 1-2*((l1+l2-l)%2);
    }

  };


}

#endif 
