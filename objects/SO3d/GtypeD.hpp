// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef _GtypeD
#define _GtypeD

//#include "Tensor.hpp"

namespace GElib{

  template<typename GROUP>
  class GtypeD: public map<typename GROUP::IrrepIx,int>{
  public:

    typedef typename GROUP::IrrepIx IrrepIx;

    GtypeD(){}

    GtypeD(const initializer_list<int>& list){
      IrrepIx l=0;
      for(auto p:list)
	(*this)[l++]=p;
    }


  public: // ---- Named constructors ------------------------------------------------------------------------


    static GtypeD Fourier(const int maxl){
      GtypeD r;
      for(int l=0; l<=maxl; l++)
	r[l]=GROUP::dim_of_irrep(l);
      return r;
    }

    static GtypeD Fourier(const initializer_list<int>& list){
      GtypeD r;
      for(auto l:list)
	r[l]=GROUP::dim_of_irrep(l);
      return r;
    }


  public: // ---- Access ------------------------------------------------------------------------------------


    IrrepIx max_irrep() const{
      IrrepIx t=0;
      for(auto p: *this)
	t=std::max(t,p.first);
      return t;
    }

    int operator()(const IrrepIx& ix) const{
      return const_cast<GtypeD&>(*this)[ix];
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "GElib::GtypeD";
    }

    string str() const{
      ostringstream oss;
      int i=0; 
      oss<<"(";
      for(auto& p:*this){
	oss<<p.first<<":"<<p.second<<",";
      }
      oss<<"\b)";
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const GtypeD& x){
      stream<<x.str(); return stream;
    }



  };


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
  

}

#endif 
