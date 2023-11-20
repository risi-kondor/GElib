// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef _GtypeE
#define _GtypeE

//#include "Tensor.hpp"
#include "Ggroup.hpp"
#include "GirrepIx.hpp"


namespace GElib{

  class GtypeE{
  public:

    //typedef map<GirrepIxWrapper,int> BASE;

    mutable map<GirrepIxWrapper,int> map;

    /*
    shared_ptr<Ggroup> G;

    GtypeE(const shared_ptr<Ggroup>& _G): 
      G(_G){}

    GtypeE(const shared_ptr<Ggroup>& _G, const initializer_list<int>& list){
      IrrepIx l=0;
      for(auto p:list)
	(*this)[l++]=p;
    }
    */

  public: // ---- Named constructors ------------------------------------------------------------------------

    /*
    static GtypeE Fourier(const int maxl){
      GtypeE r;
      for(int l=0; l<=maxl; l++)
	r[l]=GROUP::dim_of_irrep(l);
      return r;
    }

    static GtypeE Fourier(const initializer_list<int>& list){
      GtypeE r;
      for(auto l:list)
	r[l]=GROUP::dim_of_irrep(l);
      return r;
    }
    */

  public: // ---- Access ------------------------------------------------------------------------------------

    /*
    GirrepIx max_irrep() const{
      GirrepIx t=begin().first;
      for(auto p: *this)
	t=std::max(t,p.first);
      return t;
    }
    */

    int& operator[](const GirrepIx& _ix){
      GirrepIxWrapper ix(_ix.clone());
      return map[ix];
      //return BASE::operator[](ix);
    }

    int operator()(const GirrepIx& _ix) const{
      GirrepIxWrapper ix(_ix.clone());
      if(map.find(ix)==map.end()) return 0;
      //return const_cast<GtypeE&>(*this).BASE::operator[](ix);
      //return const_cast<GtypeE&>(*this).map[ix];
      return map[ix];
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "GElib::GtypeE";
    }

    string str() const{
      ostringstream oss;
      int i=0; 
      oss<<"(";
      for(auto& p:map){
	oss<<p.first<<":"<<p.second<<",";
      }
      oss<<"\b)";
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const GtypeE& x){
      stream<<x.str(); return stream;
    }

  };



  /*
  template<typename GROUP>
  typename GROUP::TAU CGproduct(const GtypeD<GROUP>& x, const GtypeD<GROUP>& y, const typename GROUP::IrrepIx lim=-1){
    GtypeD<GROUP> r;
    for(auto p1:x)
      for(auto p2:y)
	GROUP::for_each_CGcomponent(p1.first,p2.first,[&](const typename GROUP::IrrepIx& ix, const int m){
	    if(lim==-1 || ix<=lim) r[ix]+=p1.second*p2.second*m;
	  });
    return r;
  }
  
  template<typename GROUP>
  typename GROUP::TAU DiagCGproduct(const GtypeD<GROUP>& x, const GtypeD<GROUP>& y, const typename GROUP::IrrepIx lim=-1){
    GtypeD<GROUP> r;
    for(auto p1:x)
      for(auto p2:y)
	GROUP::for_each_CGcomponent(p1.first,p2.first,[&](const typename GROUP::IrrepIx& ix, const int m){
	    if(lim==-1 || ix<=lim) r[ix]+=p1.second*m;
	  });
    return r;
  }
  */

}

#endif 
