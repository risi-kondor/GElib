
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _CyclicGroup
#define _CyclicGroup

#include "Group.hpp"
#include "CyclicGroupElement.hpp"
#include "CyclicGroupIrrep.hpp"


namespace GElib{

  class CyclicGroup: public Group{
  public:

    int n;

  public:

    CyclicGroup(const int _n): n(_n){}

    static CyclicGroupElement dummy_element(){return CyclicGroupElement(1,0);}
    static CyclicGroupIrrep dummy_irrep(){return CyclicGroupIrrep(1,0);}


  public:

    int size() const{
      return n;
    }

    CyclicGroupElement identity() const{
      return CyclicGroupElement(n,0);
    }

    CyclicGroupElement element(const int i) const{
      return CyclicGroupElement(n,i);
    }

    int index(const CyclicGroupElement& x) const{
      return x.i;
    }


  public:

    int n_irreps() const{
      return n;
    }

    CyclicGroupIrrep irrep(const int i) const{
      return CyclicGroupIrrep(n,i);
    }


  public: // I/O

    string str(const string indent="") const{
      return "CyclicGroup("+to_string(n)+")";
    }

    friend ostream& operator<<(ostream& stream, const CyclicGroup& x){
      stream<<x.str(); return stream;
    }


  };

}

#endif
