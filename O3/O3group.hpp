/*
 * This file is part of GElib, a C++/CUDA library for group equivariant 
 * tensor operations. 
 *  
 * Copyright (c) 2025, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with GElib in the file NONCOMMERICAL.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in orginal
 * or modified form) must retain this copyright notice and must be 
 * accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _GElibO3group
#define _GElibO3group

#include "GElib_base.hpp"
#include "O3index.hpp"


namespace GElib{

  class O3index;
  class O3type;


  class O3group{
  public:

    typedef O3index GINDEX;
    typedef O3type GTYPE;


  public: // ---- CG-products --------------------------------------------------------------------------------


    static int CGmultiplicity(const  O3index p1, const O3index p2, const O3index p){
      int l1=p1.getl();
      int l2=p2.getl();
      int l=p.getl();
      if(l>=std::abs(l1-l2) && l<=l1+l2) return 1;
      return 0;
    }

    // TODO 
    static void for_each_CGcomponent(const O3index p1, const O3index p2, 
      const std::function<void(const O3index l, const int m)>& lambda){
      //for(int l=std::abs(l1-l2); l<=l1+l2; l++)
      //lambda(l,1);
    }

    static int CG_sign_rule(const O3index p1, const O3index p2, const O3index p){
      //return 1-2*((l1+l2-l)%2);
      return 0;
    }


  };

}


#endif 

