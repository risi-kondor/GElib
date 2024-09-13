
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3_SPHgen
#define _SO3_SPHgen

#include "TensorView.hpp"


namespace GElib{


  class SO3SPHgen{
  public:

    using TENSOR=cnine::TensorView<float>;

    int L;
    TENSOR c1;
    TENSOR c2;
    mutex mx;

    
    SO3SPHgen(): 
      L(0), 
      c1({1,1}), 
      c2({1,1}){}


    TENSOR operator()(const int _L, const float x){
      lock_guard<mutex> lock(mx);
      if(_L>L) extend(_L);

      TENSOR R({_L+1,_L+1});
      R(0,0)=sqrt(1.0/(M_PI*4.0));
      float xfact=sqrt((1.0-x)*(1.0+x));
      
      for(int l=1; l<=_L; l++){
	R(l,l)=c1(l,l)*R(l-1,l-1)*xfact;
	R(l,l-1)=c1(l,l-1)*R(l-1,l-1)*x;
	for(int m=0; m<l-1; m++)
	  R(l,m)=c1(l,m)*R(l-1,m)*x+c2(l,m)*R(l-2,m);
      }

      return R; 
   }


  private:

    void extend(const int _L){

      TENSOR newc1({_L+1,_L+1});
      for(int i=0; i<=L; i++)
	for(int j=0; j<=L; j++)
	  newc1(i,j)=c1(i,j);
      c1.reset(newc1);
      //c1=move(newc1);
	
      TENSOR newc2({_L+1,_L+1});
      for(int i=0; i<=L; i++)
	for(int j=0; j<=L; j++)
	  newc2(i,j)=c2(i,j);
      c2.reset(newc2);
      //c2=move(newc2);

      for(int l=L+1; l<=_L; l++)
	//c1(l,l)=-sqrt(((float)(2.0*l+1))/(2.0*l-1))*sqrt(((float)(2*l-2+(l==1)))/(2*l))*(2.0*l-1);
	c1(l,l)=-sqrt(((float)(2.0*l+1))/(2.0*l-1))*sqrt(1.0/(2.0*l)/(2.0*l-1))*(2.0*l-1);
      
      for(int l=L+1; l<=_L; l++)
	for(int m=0; m<l; m++){
	  float prefact0=sqrt(((float)(2.0*l+1))/(2.0*l-1));
	  float prefact1=sqrt((float)(l-m))*sqrt(1.0/(l+m));
	  c1(l,m)=prefact0*prefact1*(2.0*l-1)/(l-m);
	  if(m<l-1){
	    float prefact0b=sqrt(((float)(2.0*l+1))/(2.0*l-3));
	    float prefact2=sqrt((float)(l-m)*(l-m-1))*sqrt(1.0/((l+m)*(l+m-1.0)));
	    c2(l,m)=-prefact0b*prefact2*((float)(l+m-1.0))/(l-m);
	  }
	}

      L=_L;
    }


  };

}


#endif
