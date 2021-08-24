
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SnBank
#define _SnBank

#include "GElib_base.hpp"
#include "SnObj.hpp"
//#include "IntegerPartitionsObj.hpp"

namespace GElib{

  class SnBank{
  public:
    
    vector<SnObj*> sn;
    

  public: 

    SnObj* get_Sn(const int n){
      if(n-1<sn.size()) return sn[n-1];
      const int _n=sn.size();
      sn.resize(n-1);
      for(int i=_n+1; i<=n; i++)
	sn[i-1]=new SnObj(i);
      return sn[n-1];
    }

   
  public: 


  };
  

}

#endif 
