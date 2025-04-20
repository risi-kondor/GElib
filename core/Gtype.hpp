/*
 * This file is part of GElib, a C++/CUDA library for group equivariant 
 * tensor operations. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with GElib in the file NONCOMMERICAL.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in orginal
 * or modified form) must retain this copyright notice and must be 
 * accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _Gtype
#define _Gtype

#include "GElib_base.hpp"


namespace GElib{

  struct Gtype_type{};


  //template<typename IrrepIx>
  template<typename GROUP>
  class Gtype: public Gtype_type{
  public:

    //typedef map<GirrepIxWrapper,int> BASE;

    typedef typename GROUP::GINDEX GINDEX;
    typedef typename GROUP::GTYPE GTYPE;

    mutable map<GINDEX,int> parts;

    Gtype(){}

    Gtype(const std::map<GINDEX,int>& _parts):
      parts(_parts){}

    /*
    shared_ptr<Ggroup> G;

    Gtype(const shared_ptr<Ggroup>& _G): 
      G(_G){}

    Gtype(const shared_ptr<Ggroup>& _G, const initializer_list<int>& list){
      GINDEX l=0;
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

    GINDEX highest() const{
      return (parts.crbegin()++)->first;
    }

    int& operator[](const GINDEX& ix){
      return parts[ix];
    }

    int operator()(const GINDEX& ix) const{
      if(parts.find(ix)==parts.end()) return 0;
      return parts[ix];
    }

    void set(const GINDEX& ix, const int n){
      parts[ix]=n;
    }


  public: // ---- CG-products --------------------------------------------------------------------------------

    /*
    template<typename GTYPE>
    GTYPE CGproduct(const GTYPE& y) const{
      auto& x=static_cast<const GTYPE&>(*this);
      GTYPE R;
      for(auto& p:x.parts)
	for(auto& q:y.parts)
	  GTYPE::Group::for_each_CGcomponent(p.first,q.first,[&](const typename GTYPE::GINDEX& z, const int m){
	      R.parts[z]+=m*p.second*q.second;});
      return R;
    }
    */

    //template<typename GTYPE>
    GTYPE CGproduct(const GTYPE& y, const GINDEX& limit=GTYPE::null_ix) const{
      auto& x=static_cast<const GTYPE&>(*this);
      GTYPE R;
      for(auto& p:x.parts)
	for(auto& q:y.parts)
	  GROUP::for_each_CGcomponent(p.first,q.first,[&](const GINDEX& z, const int m){
	      if(limit==GTYPE::null_ix || z<=limit ) R.parts[z]+=m*p.second*q.second;});
      return R;
    }


  public: // ---- Diag CG-products ---------------------------------------------------------------------------


    //template<typename GTYPE>
    GTYPE DiagCGproduct(const GTYPE& y, const GINDEX& limit=GTYPE::null_ix) const{
      auto& x=static_cast<const GTYPE&>(*this);
      GTYPE R;
      for(auto& p:x.parts)
	for(auto& q:y.parts)
	  GROUP::for_each_CGcomponent(p.first,q.first,[&](const GINDEX& z, const int m){
	      if(limit==GTYPE::null_ix || z<=limit ) R.parts[z]+=m*p.second;});
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
