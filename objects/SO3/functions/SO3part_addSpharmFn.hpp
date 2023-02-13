// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3part_addSpharmFn
#define _SO3part_addSpharmFn

#include "GElib_base.hpp"
#include "CtensorB.hpp"
#include "SO3part3_view.hpp"
#include "MultiLoop.hpp"

extern GElib::SO3_SPHgen SO3_sphGen;


namespace GElib{


  class SO3part_addSpharmFn{
  public:

    void operator()(SO3part3_view& r, const cnine::Rtensor3_view& x){

      //cout<<x.n0<<" "<<x.n1<<" "<<x.n2<<" "<<x.s0<<" "<<x.s1<<" "<<x.s2<<endl;

      CNINE_ASSRT(r.dev==0);
      CNINE_ASSRT(x.dev==0);
      int l=r.getl();
      int B=r.n0;
      int n=r.n2;
      //cnine::Ctensor3_view v=view3();
      //assert(x.dims.size()==3);
      assert(x.n0==B);
      assert(x.n1==3);
      assert(x.n2==n);

      for(int b=0; b<B; b++){
	for(int j=0; j<n; j++){
	  float vx=x(b,0,j);
	  float vy=x(b,1,j);
	  float vz=x(b,2,j);
	  float length=sqrt(vx*vx+vy*vy+vz*vz); 
	  float len2=sqrt(vx*vx+vy*vy);

	  //cout<<len2<<endl;
	  if(len2==0 || std::isnan(vx/len2) || std::isnan(vy/len2)){
	    float a=sqrt(((float)(2*l+1))/(M_PI*4.0));
	    for(int j=0; j<n; j++)
	      r.inc(b,0,j,a);

	  }else{

	    complex<float> cphi(vx/len2,vy/len2);
	    cnine::Gtensor<float> P=SO3_sphGen(l,vz/length);
	    vector<complex<float> > phase(l+1);
	    phase[0]=complex<float>(1.0,0);
	    for(int m=1; m<=l; m++)
	      phase[m]=cphi*phase[m-1];
	    
	    for(int m=0; m<=l; m++){
	      complex<float> a=phase[m]*complex<float>(P(l,m)); // *(1-2*(m%2))
	      complex<float> aa=complex<float>(1-2*(m%2))*std::conj(a);
	      r.inc(b,m,j,a);
	      //cout<<b<<" "<<l+m<<" "<<j<<" "<<a<<endl;
	      if(m>0) r.inc(b,-m,j,aa);
	    }
	  }

	}

      }

    }

  };

}


#endif 
