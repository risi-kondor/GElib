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

#include "GElib_base.hpp" 
#include "SO3_CGindex.hpp" 
#include "SO3_CGcoeffs.hpp" 


namespace GElib{



  template<class TYPE>
  class SO3_CGcoeffs: public cnine::TensorView<TYPE>{ 
  public:

    typedef cnine::TensorView<TYPE> TENSOR;


    int l,l1,l2;

    SO3_CGcoeffs(const CGindex& ix):
      TENSOR({2*ix.l1+1,2*ix.l2+1},0,0), 
      l(ix.l), l1(ix.l1), l2(ix.l2){
      for(int m1=-l1; m1<=l1; m1++)
	for(int m2=std::max(-l2,-l-m1); m2<=std::min(l2,l-m1); m2++)
	  (*this)(m1,m2)=slowCG(m1,m2);
    }

    //SO3_CGcoeffs(Gtensor<TYPE>&& T, const int _l, const int _l1, const int _l2): 
    //Gtensor<TYPE>(std::move(T)), l(_l), l1(_l1), l2(_l2){}
    
    ~SO3_CGcoeffs(){
    }
    

  public:
    
    //SO3_CGcoeffs(const SO3_CGcoeffs<TYPE>&  x): 
    //cnine::Gtensor<TYPE>(x,cnine::nowarn), l(x.l), l1(x.l1), l2(x.l2){};

    //SO3_CGcoeffs& operator=(const SO3_CGcoeffs<TYPE>& x)=delete;
    
  public:

    //TYPE& element(const int m1, const int m2){
    //return (*this)(m1+l1,m2+l2);
    //}


  private:

    double logfact(double n){
      return lgamma(n+1);
    }
    
    double plusminus(int k){ if(k%2==1) return -1; else return +1; }

    double slowCG(const int m1, const int m2){
      
      int m=m1+m2;
      int m3=-m;
      int t1=l2-m1-l;
      int t2=l1+m2-l;
      int t3=l1+l2-l;
      int t4=l1-m1;
      int t5=l2+m2;
  
      int tmin=std::max(0,std::max(t1,t2));
      int tmax=std::min(t3,std::min(t4,t5));

      double logA=(logfact(l1+l2-l)+logfact(l1-l2+l)+logfact(-l1+l2+l)-logfact(l1+l2+l+1))/2;
      logA+=(logfact(l1+m1)+logfact(l1-m1)+logfact(l2+m2)+logfact(l2-m2)+logfact(l+m3)+logfact(l-m3))/2;

      double wigner=0;
      for(int t=tmin; t<=tmax; t++){
	double logB=logfact(t)+logfact(t-t1)+logfact(t-t2)+logfact(t3-t)+logfact(t4-t)+logfact(t5-t);
	wigner += plusminus(t)*exp(logA-logB);
      }
      
      return plusminus(l1-l2-m3)*plusminus(l1-l2+m)*sqrt((double)(2*l+1))*wigner; 
    }

  };

  
} 


#endif
