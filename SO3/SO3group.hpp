/*
 * This file is part of GElib, a C++/CUDA library for group equivariant 
 * tensor operations. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with GElib in the file NONCOMMERICAL.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in orginal
 * or modified form) must retain this copyright notice and must be 
 * accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _GElibSO3group
#define _GElibSO3group

#include "GElib_base.hpp"
#include "Gtype.hpp"


namespace GElib{

  class SO3group{
  public:

    typedef int IrrepIx;


  public: // ---- CG-products --------------------------------------------------------------------------------


    static int CGmultiplicity(const int l1, const int l2, const int l){
      if(l>=std::abs(l1-l2) && l<=l1+l2) return 1;
      return 0;
    }

    static void for_each_CGcomponent(const int l1, const int l2, 
      const std::function<void(const int l, const int m)>& lambda){
      for(int l=std::abs(l1-l2); l<=l1+l2; l++)
	lambda(l,1);
    }

    static int CG_sign_rule(const int l1, const int l2, const int l){
      return 1-2*((l1+l2-l)%2);
    }

  };

}


#endif 

