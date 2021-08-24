
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3element
#define _SO3element

#include "GElib_base.hpp"
//#include "R3vector.hpp"

extern default_random_engine rndGen;

namespace GElib{

  class SO3element{
  public:

    double phi, psi, theta;

    SO3element(const double _phi, const double _theta, const double _psi):
      phi(_phi), psi(_psi), theta(_theta){}

    //template<typename TYPE>
    //SO3element(const R3vector<TYPE>& x){
    //if(x[1]!=0) phi=atan(x[0]/x[1]); else phi=0;
    //psi=0;
    //theta=asin(x[2]/sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]));
    //}

    SO3element(const cnine::fill_uniform& dummy){
      uniform_real_distribution<double> distr;
      phi=distr(rndGen)*M_PI*2;
      theta=distr(rndGen)*M_PI;
      psi=distr(rndGen)*M_PI*2;
    }


    // ---- Operations ---------------------------------------------------------------------------------------


    cnine::Gtensor<float> operator()(const cnine::Gtensor<float>& x){
      assert(x.k==1);
      assert(x.dims[0]==3);
      cnine::Gtensor<float> R({3},cnine::fill::raw);
      float a=cos(psi)*x(0)+sin(psi)*x(1);
      float b=-sin(psi)*x(0)+cos(psi)*x(1);
      float c=x(2);
      //float d=a;
      //float e=cos(theta)*b+sin(theta)*c;
      //float f=-sin(theta)*b+cos(theta)*c;
      float d=cos(theta)*a+sin(theta)*c;
      float e=b;
      float f=-sin(theta)*a+cos(theta)*c;
      R(0)=cos(phi)*d+sin(phi)*e;
      R(1)=-sin(phi)*d+cos(phi)*e;
      R(2)=f;
      return R;
    }


    // ---- I/O ----------------------------------------------------------------------------------------------


    string str() const{
      return "("+to_string(phi)+","+to_string(theta)+","+to_string(psi)+")";
    }

    friend ostream& operator<<(ostream& stream, const GElib::SO3element& x){
      stream<<x.str(); return stream;}

  };

}



#endif

