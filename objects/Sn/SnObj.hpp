
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SnObj
#define _SnObj

#include "GElib_base.hpp"

#include "SnElement.hpp"


namespace GElib{

  class SnObj{
  public:

    const int n;
    int order;

    // _Sn* S_nm;

    SnObj(const int _n): n(_n){
      order=factorial(n);
      cout<<"Creating Sn("<<n<<")"<<endl;
    }


  public:

    SnElement element(int e){
      SnElement p(n,cnine::fill_identity());
      vector<int> r(n);
      for(int i=0; i<n; i++) r[i]=i+1;

      vector<int> s(n);
      for(int i=n; i>0; i--){
	s[i-1]=i-e/factorial(i-1);
	e=e%factorial(i-1);
      }

      for(int i=2; i<=n; i++){
	int t=s[i-1];
	for(int k=i; k>=t+1; k--) 
	  p[k-1]=p[k-2];
	p[t-1]=i;
      }

      return p;
    }

  };

}

#endif
