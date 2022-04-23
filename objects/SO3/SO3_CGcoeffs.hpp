
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3_CGcoeffs
#define _SO3_CGcoeffs

#include "SO3_CGindex.hpp" 
#include "SO3_CGcoeffs.hpp" 
#include "SO3_CGbank.hpp" 
#include "Gtensor.hpp"

#define _SO3_CGcoeff_TYPE double 

extern default_random_engine rndGen;


namespace GElib{


  //template<class TYPE>
  //using Gtensor=cnine::Gtensor<TYPE>;


  template<class TYPE>
  class SO3_CGcoeffs: public cnine::Gtensor<TYPE>{ 
  public:

    template<class TYPE2>
    using Gtensor=cnine::Gtensor<TYPE2>;

    using Gtensor<TYPE>::arr; 
    using Gtensor<TYPE>::arrg; 

    int l,l1,l2;

    SO3_CGcoeffs(const CGindex& ix):
      Gtensor<TYPE>({2*ix.l1+1,2*ix.l2+1},cnine::fill::zero,0), 
      l(ix.l), l1(ix.l1), l2(ix.l2){
      for(int m1=-l1; m1<=l1; m1++)
	for(int m2=std::max(-l2,-l-m1); m2<=std::min(l2,l-m1); m2++)
	  this->element(m1,m2)=slowCG(m1,m2);
    }

    SO3_CGcoeffs(Gtensor<TYPE>&& T, const int _l, const int _l1, const int _l2): 
      Gtensor<TYPE>(std::move(T)), l(_l), l1(_l1), l2(_l2){}
    
    ~SO3_CGcoeffs(){
    }
    

  public:
    
    SO3_CGcoeffs(const SO3_CGcoeffs<TYPE>&  x): 
      cnine::Gtensor<TYPE>(x,cnine::nowarn), l(x.l), l1(x.l1), l2(x.l2){};

    SO3_CGcoeffs& operator=(const SO3_CGcoeffs<TYPE>& x)=delete;
    
  public:

    TYPE& element(const int m1, const int m2){
      return (*this)(m1+l1,m2+l2);
    }


  private:

    _SO3_CGcoeff_TYPE logfact(int n){
      return lgamma((_SO3_CGcoeff_TYPE)(n+1));
    }
    
    _SO3_CGcoeff_TYPE plusminus(int k){ if(k%2==1) return -1; else return +1; }

    _SO3_CGcoeff_TYPE slowCG(const int m1, const int m2){
      
      int m=m1+m2;
      int m3=-m;
      int t1=l2-m1-l;
      int t2=l1+m2-l;
      int t3=l1+l2-l;
      int t4=l1-m1;
      int t5=l2+m2;
  
      int tmin=std::max(0,std::max(t1,t2));
      int tmax=std::min(t3,std::min(t4,t5));

      _SO3_CGcoeff_TYPE logA=(logfact(l1+l2-l)+logfact(l1-l2+l)+logfact(-l1+l2+l)-logfact(l1+l2+l+1))/2;
      logA+=(logfact(l1+m1)+logfact(l1-m1)+logfact(l2+m2)+logfact(l2-m2)+logfact(l+m3)+logfact(l-m3))/2;

      _SO3_CGcoeff_TYPE wigner=0;
      for(int t=tmin; t<=tmax; t++){
	_SO3_CGcoeff_TYPE logB=logfact(t)+logfact(t-t1)+logfact(t-t2)+logfact(t3-t)+logfact(t4-t)+logfact(t5-t);
	wigner += plusminus(t)*exp(logA-logB);
      }
      
      return plusminus(l1-l2-m3)*plusminus(l1-l2+m)*sqrt((_SO3_CGcoeff_TYPE)(2*l+1))*wigner; 
    }

  };

  
} 


#endif
