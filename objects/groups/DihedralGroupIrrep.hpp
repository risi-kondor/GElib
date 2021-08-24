
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _DihedralGroupIrrep
#define _DihedralGroupIrrep

#include "CtensorObj.hpp"

#include "Group.hpp"
#include "DihedralGroupElement.hpp"

namespace GElib{

  class DihedralGroupIrrep: public GroupIrrep{
  public:

    int n;
    int k;

    typedef cnine::Gdims Gdims;
    typedef cnine::CtensorObj ctensor;


  public:

    DihedralGroupIrrep(const int _n, const int _k): 
      n(_n), k(_k){}

  public:

    int dim() const {
      if(k<2*(1+(n%2==0))) return 1;
      else return 2;
    }

    ctensor operator()(const int i) const{

      if(k==0){
	return ctensor(Gdims({1}),cnine::fill_ones());
      }

      if(k==1){
	ctensor r(Gdims({1}),cnine::fill_ones());
	r.set_value(0,1.0-2.0*(i>=n));
	return r;
      }

      if(n%2==0){
	if(k==2){
	  ctensor r(Gdims({1}),cnine::fill_ones());
	  r.set_value(0,1.0-2.0*(i%2));
	  return r;
	}
	if(k==2){
	  ctensor r(Gdims({1}),cnine::fill_ones());
	  r.set_value(0,(1.0-2.0*(i%2))*(1.0-2.0*(i%2)));
	  return r;
	}
      }

      ctensor r(Gdims({2,2}),cnine::fill_zero());
      const int p=k-1-2*(n%2==0);
      const float c=2*M_PI/n;
      const complex<float> a=complex<float>(cos(c*p*i),sin(c*p*i));
      const complex<float> b=complex<float>(cos(c*p*i),-sin(c*p*i));
      if(i<n){
	r.set_value(0,0,a);
	r.set_value(1,1,b);
      }else{
	r.set_value(0,1,a);
	r.set_value(1,0,b);
      }      
      return r;
    }

    ctensor operator()(const DihedralGroupElement& x) const{
      return operator()(x.index());
    }


  public: // I/O

    string str(const string indent="") const{
      return "DihedralGroupIrrep("+to_string(n)+","+to_string(k)+")";
    }

    friend ostream& operator<<(ostream& stream, const DihedralGroupIrrep& x){
      stream<<x.str(); return stream;
    }


  };

}

#endif
