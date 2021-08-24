
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _DihedralGroupElement
#define _DihedralGroupElement

#include "GElib_base.hpp"
#include "Group.hpp"


namespace GElib{

  class DihedralGroupElement: public GroupElement{
  public:

    int n;
    int i;
    int s=1;

  public:

    DihedralGroupElement(const int _n, const cnine::fill_identity& dummy): 
      n(_n), i(0), s(1){}

    DihedralGroupElement(const int _n, const int _i, const int _s=1): 
      n(_n), i(_i), s(_s){}

  public:

    int index() const{
      if(s==1) return i;
      else return n+i;
    }

    DihedralGroupElement inverse() const{
      return DihedralGroupElement(n,(s*(-i)+n)%n,s);
    }

    DihedralGroupElement operator*(const DihedralGroupElement& y) const{
      assert(y.n==n);
      return DihedralGroupElement(n,(i+s*y.i+n)%n,s*y.s);
    }


  public: // I/O

    string str(const string indent="") const{
      return "Dihedral("+to_string(n)+","+to_string(i)+","+to_string(s)+")";
    }

    friend ostream& operator<<(ostream& stream, const DihedralGroupElement& x){
      stream<<x.str(); return stream;
    }

  };

}

#endif
