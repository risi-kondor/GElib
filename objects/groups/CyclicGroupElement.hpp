
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _CyclicGroupElement
#define _CyclicGroupElement

#include "GElib_base.hpp"
#include "Group.hpp"


namespace GElib{

  class CyclicGroupElement: public GroupElement{
  public:

    int n;
    int i;

  public:

    CyclicGroupElement(const int _n, const cnine::fill_identity& dummy): 
      n(_n), i(0){}

    CyclicGroupElement(const int _n, const int _i): 
      n(_n), i(_i){}

  public:

    int index() const{
      return i;
    }

    CyclicGroupElement inverse() const{
      return CyclicGroupElement(n,(-i+n)%n);
    }

    CyclicGroupElement operator*(const CyclicGroupElement& y) const{
      assert(y.n==n);
      return CyclicGroupElement(n,(i+y.i)%n);
    }


  public: // I/O

    string str(const string indent="") const{
      return "Cyclic("+to_string(n)+","+to_string(i)+")";
    }

    friend ostream& operator<<(ostream& stream, const CyclicGroupElement& x){
      stream<<x.str(); return stream;
    }

  };

}

#endif
