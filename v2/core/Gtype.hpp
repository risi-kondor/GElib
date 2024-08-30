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


  template<typename IrrepIx>
  class Gtype{
  public:

    //typedef map<GirrepIxWrapper,int> BASE;

    mutable map<IrrepIx,int> map;

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
	r[l]=GROUP::dim_of_irrep(l);
      return r;
    }

    static Gtype Fourier(const initializer_list<int>& list){
      Gtype r;
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

    int& operator[](const IrrepIx& ix){
      return map[ix];
    }

    int operator()(const IrrepIx& ix) const{
      if(map.find(ix)==map.end()) return 0;
      return map[ix];
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "GElib::Gtype";
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

    friend ostream& operator<<(ostream& stream, const Gtype& x){
      stream<<x.str(); return stream;
    }

  };




}

#endif 
