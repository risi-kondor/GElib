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
#include "TensorView.hpp"
#include "MultiLoop.hpp"
#include "TensorUtils.hpp"
#include "SO3SPHgen.hpp"


namespace GElib{

extern GElib::SO3SPHgen SO3_SPHgen; // Moved into namespace GElib


  template<typename TYPE>
  class SO3part_addSpharmFn{
  public:

    using TENSOR=cnine::TensorView<TYPE>;
    using CTENSOR=cnine::TensorView<complex<TYPE> >;


    template<typename GPART>
    void operator()(GPART& r, const TENSOR& x){
      GELIB_ASSRT(x.ndims()>=3);
      int l=r.getl();
      int nc=r.getn();
      GELIB_ASSRT(x.dims(-1)==nc);

      if(r.get_dev()==0){
	r.template for_each_cell_multi<TYPE>(x,[&](const int b, const int g, const CTENSOR& r, const TENSOR& x){
	    for(int j=0; j<nc; j++){

	      float vx=x(0,j);
	      float vy=x(1,j);
	      float vz=x(2,j);
	      float length=sqrt(vx*vx+vy*vy+vz*vz); 
	      float len2=sqrt(vx*vx+vy*vy);

	      if(len2==0 || std::isnan(vx/len2) || std::isnan(vy/len2)){
		float a=sqrt(((float)(2*l+1))/(M_PI*4.0));
		r.inc(0,j,a); // check this
	      }else{
		complex<float> cphi(vx/len2,vy/len2);
		auto P=SO3_SPHgen(l,vz/length);
		vector<complex<float> > phase(l+1);
		phase[0]=complex<float>(1.0,0);
		for(int m=1; m<=l; m++)
		  phase[m]=cphi*phase[m-1];
	    
		for(int m=0; m<=l; m++){
		  complex<float> a=phase[m]*complex<float>(P(l,m)); // *(1-2*(m%2))
		  complex<float> aa=complex<float>(1-2*(m%2))*std::conj(a);
		  r.inc(l+m,j,a);
		  if(m>0) r.inc(l-m,j,aa);
		}
	      }
	    }
	  });
      }// dev==0
 
   }

  };

}


#endif 
