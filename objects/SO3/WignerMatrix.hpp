
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _WignerMatrix
#define _WignerMatrix

#include "Gtensor.hpp"
#include "SO3element.hpp"
//#include "Factorial.hpp"
#include "Ctensor2_view.hpp"

extern default_random_engine rndGen;

namespace GElib{


  template<class TYPE>
  class WignerMatrix: public cnine::Gtensor<complex<TYPE> >{ 
  public:

    template<class TYPE2>
    using Gtensor=cnine::Gtensor<TYPE2>;

    WignerMatrix(){}

    WignerMatrix(const int l, const SO3element& x): 
      WignerMatrix(l,x.phi,x.theta,x.psi){}

    WignerMatrix(const int l, const double phi, const double theta, const double psi): 
      Gtensor<complex<TYPE> >({2*l+1,2*l+1},cnine::fill::raw){
      for(int m1=-l; m1<=l; m1++)
	for(int m2=-l; m2<=l; m2++){
	  complex<TYPE> d=littled(l,m2,m1,theta);
	  (*this)(m1+l,m2+l)=d*exp(-complex<TYPE>(0,m1*phi))*exp(-complex<TYPE>(0,m2*psi));
	}
    }

    WignerMatrix(const int l, const cnine::fill_uniform& dummy): 
      Gtensor<complex<TYPE> >(2*l+1,2*l+1){
      uniform_real_distribution<TYPE> distr;
      *this=WignerMatrix(l, distr(rndGen)*M_PI*2,distr(rndGen)*M_PI,distr(rndGen)*M_PI*2);
    }

    
  public:

    static TYPE littled(const int l, const int m1, const int m2, const double beta){
      double x=0;

      if(l<5){
	for(int s=std::max(0,m1-m2); s<=std::min(l+m1,l-m2); s++){
	  TYPE pref=1.0/(cnine::factorial(l+m1-s)*cnine::factorial(s)*cnine::factorial(m2-m1+s)*cnine::factorial(l-m2-s));
	  if((m2-m1+s)%2) pref=-pref;
	  x+=pref*std::pow(cos(beta/2),2*l+m1-m2-2*s)*std::pow(sin(beta/2),m2-m1+2*s);
	}
	TYPE v= sqrt(cnine::factorial(l+m1)*cnine::factorial(l-m1)*cnine::factorial(l+m2)*cnine::factorial(l-m2))*x;
	if(std::isnan(v)) cout<<l<<m1<<m2<<" "<<beta<<endl;
      return v;
      }

      // check this!
      for(int s=std::max(0,m1-m2); s<=std::min(l+m1,l-m2); s++){
	double a=(lgamma(l+m1+1)+lgamma(l-m1+1)+lgamma(l+m2+1)+lgamma(l-m2+1))/2.0;
	a-=lgamma(l+m1-s+1)+lgamma(s+1)+lgamma(m2-m1+s+1)+lgamma(l-m2-s+1);
	if(std::isnan(std::exp(a))) cout<<s<<" "<<l<<m1<<m2<<" "<<beta<<endl;
	x+=(1-2*((m2-m1+s)%2))*std::pow(cos(beta/2),2*l+m1-m2-2*s)*std::pow(sin(beta/2),m2-m1+2*s)*std::exp(a);
	if(std::isnan(x)){
	  cout<<l<<m1<<m2<<" "<<beta<<" ww "<<a<<" "<<std::exp(a)<<" "<<x<<" ";
	  cout<<(2*(s%2)-1)*std::pow(cos(beta/2),2*l+m1-m2-2*s)*std::pow(sin(beta/2),m2-m1+2*s)<<endl;
	}
      }
      if(std::isnan(x)) cout<<l<<m1<<m2<<" "<<beta<<endl;
      return x;
    }


  };


  inline void add_WignerMatrix_to(const cnine::Ctensor2_view& x, 
    const int l, const double phi, const double theta, const double psi){
    assert(l>=0);
    assert(x.n0==2*l+1);
    assert(x.n1==2*l+1);

    for(int m1=-l; m1<=l; m1++)
      for(int m2=-l; m2<=l; m2++){
	complex<float> d=WignerMatrix<float>::littled(l,m2,m1,theta);
	x.inc(m1+l,m2+l,d*exp(-complex<float>(0,m1*phi))*exp(-complex<float>(0,m2*psi)));
      }
  }

  inline void add_WignerMatrix_to(cnine::CtensorB& x, 
    const int l, const double phi, const double theta, const double psi){
    add_WignerMatrix_to(x.view2(),l,phi,theta,psi);
  }

  //inline Gtensor<float> SO3element::operator()(const Gtensor<float>& x){
  //auto D=WignerMatrix<float>(1,*this);
  //cout<<D<<endl; 
  //return D*x;
  //}


}

#endif
