// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef _SO3
#define _SO3

namespace GElib{

  class SO3{
  public:

    typedef int IrrepIx;
  
  

  public: // ---- CG-rule -----------------------------------------------------------------------------------


    static void for_each_CGcomponent(const int l1, const int l2, 
      const std::function<void(const int&, const int)>& lambda){
      for(int l=std::abs(l1-l2); l<=l1+l2; l++)
	lambda(l,1);
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------

    static string repr(){
      return "SO3";
    }

  };

}

#endif 
