// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef _Gtype
#define _Gtype

#include "GElib_base.hpp"


namespace GElib{

  struct Gtype_type{};


  template<typename IrrepIx>
  class Gtype: public Gtype_type{
  public:

    //typedef map<GirrepIxWrapper,int> BASE;

    mutable map<IrrepIx,int> parts;

    Gtype(){}

    Gtype(const std::map<IrrepIx,int>& _parts):
      parts(_parts){}

    /*
    shared_ptr<Ggroup> G;

    Gtype(const shared_ptr<Ggroup>& _G): 
      G(_G){}

    Gtype(const shared_ptr<Ggroup>& _G, const initializer_list<int>& list){
      IrrepIx l=0;
      for(auto p:list)
	(*this)[l++]=p;
    }
    */

  public: // ---- Named constructors ------------------------------------------------------------------------

    /*
    static Gtype Fourier(const int maxl){
      Gtype r;
      for(int l=0; l<=maxl; l++)
	r[l]=Group::dim_of_irrep(l);
      return r;
    }

    static Gtype Fourier(const initializer_list<int>& list){
      Gtype r;
      for(auto l:list)
	r[l]=Group::dim_of_irrep(l);
      return r;
    }
    */

  public: // ---- Access ------------------------------------------------------------------------------------


    int size() const{
      return parts.size();
    }

    IrrepIx highest() const{
      return (parts.crbegin()++)->first;
    }

    int& operator[](const IrrepIx& ix){
      return parts[ix];
    }

    int operator()(const IrrepIx& ix) const{
      if(parts.find(ix)==parts.end()) return 0;
      return parts[ix];
    }

    void set(const IrrepIx& ix, const int n){
      parts[ix]=n;
    }


  public: // ---- CG-products --------------------------------------------------------------------------------


    template<typename GTYPE>
    GTYPE CGproduct(const GTYPE& y) const{
      auto& x=static_cast<const GTYPE&>(*this);
      GTYPE R;
      for(auto& p:x.parts)
	for(auto& q:y.parts)
	  GTYPE::Group::for_each_CGcomponent(p.first,q.first,[&](const typename GTYPE::IRREP_IX& z, const int m){
	      R.parts[z]+=m*p.second*q.second;});
      return R;
    }

    template<typename GTYPE>
    GTYPE CGproduct(const GTYPE& y, const typename GTYPE::IRREP_IX& limit) const{
      auto& x=static_cast<const GTYPE&>(*this);
      GTYPE R;
      for(auto& p:x.parts)
	for(auto& q:y.parts)
	  GTYPE::Group::for_each_CGcomponent(p.first,q.first,[&](const typename GTYPE::IRREP_IX& z, const int m){
	      if(z<=limit) R.parts[z]+=m*p.second*q.second;});
      return R;
    }


  public: // ---- Diag CG-products ---------------------------------------------------------------------------


    template<typename GTYPE>
    GTYPE DiagCGproduct(const GTYPE& y, const typename GTYPE::IRREP_IX& limit=GTYPE::null_ix) const{
      auto& x=static_cast<const GTYPE&>(*this);
      GTYPE R;
      for(auto& p:x.parts)
	for(auto& q:y.parts)
	  GTYPE::Group::for_each_CGcomponent(p.first,q.first,[&](const typename GTYPE::IRREP_IX& z, const int m){
	      if(limit==GTYPE::null_ix || z<=limit) R.parts[z]+=m*p.second;});
      return R;
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "GElib::Gtype";
    }

    string str() const{
      ostringstream oss;
      int i=0; 
      oss<<"(";
      for(auto& p:parts){
	oss<<p.first<<":"<<p.second<<",";
      }
      oss<<"\b)";
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Gtype& x){
      stream<<x.str(); return stream;
    }

  };




}

#endif 
