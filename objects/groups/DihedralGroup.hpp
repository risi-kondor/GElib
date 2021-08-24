
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _DihedralGroup
#define _DihedralGroup

#include "Group.hpp"
#include "DihedralGroupElement.hpp"
#include "DihedralGroupIrrep.hpp"


namespace GElib{

  class DihedralGroup: public Group{
  public:

    int n;

    static DihedralGroupElement dummy_element(){return DihedralGroupElement(1,0);}
    static DihedralGroupIrrep dummy_irrep(){return DihedralGroupIrrep(1,0);}


  public:

    DihedralGroup(const int _n): n(_n){}

  public:

    int size() const{
      return 2*n;
    }

    DihedralGroupElement identity() const{
      return DihedralGroupElement(n,0,1);
    }

    DihedralGroupElement element(const int i) const{
      if(i<n) return DihedralGroupElement(n,i,1);
      else return DihedralGroupElement(n,i-n,-1);
    }

    DihedralGroupElement r(const int i) const{
      return DihedralGroupElement(n,i,1);
    }

    DihedralGroupElement s() const{
      return DihedralGroupElement(n,1,-1);
    }

    int index(const DihedralGroupElement& x) const{
      return x.i+n*(x.s==-1);
    }



  public:

    int n_irreps() const{
      if(n%2==0) return n/2+3; 
      else return (n-1)/2+2;
    }

    DihedralGroupIrrep irrep(const int i) const{
      return DihedralGroupIrrep(n,i);
    }


  public: // I/O

    string str(const string indent="") const{
      return "DihedralGroup("+to_string(n)+")";
    }

    friend ostream& operator<<(ostream& stream, const DihedralGroup& x){
      stream<<x.str(); return stream;
    }


  };

}

#endif
