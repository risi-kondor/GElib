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

#ifndef _GElibO3irrep
#define _GElibO3irrep

#include "GElib_base.hpp"
#include "O3element.hpp"
#include "O3index.hpp"


namespace GElib{

  class O3irrep{
  public:

    O3index ix;
    

    O3irrep(const O3index& _ix): ix(_ix){}


  public: // ---- Operations ---------------------------------------------------------------------------------


    template<typename TYPE>
    cnine::TensorView<complex<TYPE> > matrix(const O3element<TYPE>& R) const{
      int l=ix.getl();
      cnine::TensorView<complex<TYPE> > M({2*l+1,2*l+1},0,0);
      double beta=acos(R(2,2));
      double alpha=atan2(R(2,0),-R(2,1));
      double gamma=atan2(R(0,2),R(1,2));
      //double psi=atan2(R(1,0),R(0,0));
      //double phi=atan2(R(2,1),R(2,2));
      //double theta=atan2(-R(2,0),sqrt(R(0,0)*R(0,0)+R(1,0)*R(1,0)));
      for(int m1=-l; m1<=l; m1++)
	for(int m2=-l; m2<=l; m2++){
	  //complex<TYPE> d=littled(m2,m1,theta);
	  complex<TYPE> d=littled(m2,m1,beta);
	  //M.set(m1+l,m2+l,d*exp(-complex<TYPE>(0,m1*phi))*exp(-complex<TYPE>(0,m2*psi)));
	  //M.set(m1+l,m2+l,d*exp(-complex<TYPE>(0,m1*alpha))*exp(-complex<TYPE>(0,m2*gamma)));
	  M.set(m2+l,m1+l,d*exp(-complex<TYPE>(0,m1*alpha))*exp(-complex<TYPE>(0,m2*gamma))); // why transpose?
	}
      return M;
    }
    
    
    float littled(const int m1, const int m2, const double beta) const{
      int l=ix.getl();
      double x=0;

      if(l<5){
	for(int s=std::max(0,m1-m2); s<=std::min(l+m1,l-m2); s++){
	  double pref=1.0/(cnine::factorial(l+m1-s)*cnine::factorial(s)*cnine::factorial(m2-m1+s)*cnine::factorial(l-m2-s));
	  if((m2-m1+s)%2) pref=-pref;
	  x+=pref*std::pow(cos(beta/2),2*l+m1-m2-2*s)*std::pow(sin(beta/2),m2-m1+2*s);
	}
	double v= sqrt(cnine::factorial(l+m1)*cnine::factorial(l-m1)*cnine::factorial(l+m2)*cnine::factorial(l-m2))*x;
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


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "GElib::O3irrep";
    }

    string repr() const{
      return "<O3irrep ix="+ix.str()+">";
    }

    string str(const string indent="") const{
      return "";
    }

    friend ostream& operator<<(ostream& stream, const O3irrep& x){
      stream<<x.str(); return stream;
    }


  };


}


#endif 
