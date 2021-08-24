
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _ProductGroup
#define _ProductGroup

#include "Group.hpp"
#include "ProductGroupElement.hpp"
#include "ProductGroupIrrep.hpp"


namespace GElib{

  template<typename G1, typename G2>
  class ProductGroup: public Group{
  public:

    typedef decltype(G1::dummy_element()) ELEMENT1;
    typedef decltype(G1::dummy_irrep()) IRREP1; 

    typedef decltype(G2::dummy_element()) ELEMENT2;
    typedef decltype(G2::dummy_irrep()) IRREP2; 

    const G1& g1;
    const G2& g2;

    
  public:

    ProductGroup(const G1& _g1, const G2& _g2): 
      g1(_g1), g2(_g2){}

    static ProductGroupElement<ELEMENT1,ELEMENT2> dummy_element(){
      return ProductGroupElement<ELEMENT1,ELEMENT2>(G1::dummy_element(),G2::dummy_element());}

    static ProductGroupIrrep<IRREP1,IRREP2> dummy_irrep(){
      return ProductGroupIrrep<IRREP1,IRREP2>(G1::dummy_irrep(),G2::dummy_irrep());}

      
  public:

    int size() const{
      return g1.size()*g2.size();
    }

    ProductGroupElement<decltype(g1.identity()),decltype(g2.identity())> identity() const{
      return gpair(g1.identity(),g2.identity());
    }

    ProductGroupElement<decltype(g1.identity()),decltype(g2.identity())> element(const int i) const{
      return gpair(g1.element(i/g2.size()),g2.element(i%g2.size()));
    }


  public:

    int n_irreps() const{
      return g1.n_irreps()*g2.n_irreps();
    }

    ProductGroupIrrep<IRREP1,IRREP2> irrep(const int i) const{
      //return ProductGroupIrrep<IRREP1,IRREP2>();
      return g1.irrep(i/g2.n_irreps())*g2.irrep(i%g2.n_irreps());
    }


  public: // I/O

    string str(const string indent="") const{
      return "Product<"+g1.str(),","+g2.str()+">";
    }

    friend ostream& operator<<(ostream& stream, const ProductGroup& x){
      stream<<x.str(); return stream;
    }


  };


  template<typename G1, typename G2, 
	   typename = typename std::enable_if<std::is_base_of<Group,G1>::value, G1>::type>
  ProductGroup<G1,G2> operator*(const G1& g1, const G2& g2){
    return ProductGroup<G1,G2>(g1,g2);
  }


}

#endif
