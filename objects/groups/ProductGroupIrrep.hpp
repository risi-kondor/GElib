
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _ProductGroupIrrep
#define _ProductGroupIrrep

#include "CtensorObj.hpp"

#include "Group.hpp"
#include "ProductGroupElement.hpp"

namespace GElib{


  template<typename RHO1, typename RHO2>
  class ProductGroupIrrep: public GroupIrrep{
  public:

    const RHO1 rho1;
    const RHO2 rho2;
    
    const int d;

    typedef cnine::Gdims Gdims;
    typedef cnine::CtensorObj ctensor;


  public:

    ProductGroupIrrep(const RHO1& _rho1, const RHO2& _rho2): 
      rho1(_rho1), rho2(_rho2), d(_rho1.dim()*_rho2.dim()){}

  public:

    int dim() const {return d;}

    ctensor operator()(const int i) const{
      ctensor r(Gdims({d,d}),cnine::fill_zero());
      return r;
    }

    template<typename ELEMENT1, typename ELEMENT2>
    ctensor operator()(const ProductGroupElement<ELEMENT1,ELEMENT2>& x) const{
      return operator()(x.index());
    }


  public: // I/O

    string str(const string indent="") const{
      return "Product<"+rho1.str()+","+rho2.str()+">";
    }

    friend ostream& operator<<(ostream& stream, const ProductGroupIrrep& x){
      stream<<x.str(); return stream;
    }

  };


  template<typename R1, typename R2, 
	   typename = typename std::enable_if<std::is_base_of<GroupIrrep,R1>::value, R1>::type>
  ProductGroupIrrep<R1,R2> operator*(const R1& r1, const R2& r2){
    return ProductGroupIrrep<R1,R2>(r1,r2);
  }

}

#endif
