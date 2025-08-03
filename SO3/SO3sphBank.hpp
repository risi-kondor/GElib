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

#ifndef _GElibAssocLegendre
#define _GElibAssocLegendre

namespace GElib{

  class AssoceLegendre{
  public:

    using TENSOR=cnine::TensorView<float>;

    int maxL;
    TENSOR C;
    unordered_map<int,TENSOR> remotes; 

    SO3sphBank():
      C({2,1},0,0){
      maxL=0;
    }


    // -----------------------------------------------------------------------------------------------------


    TENSOR operator()(const int L, const float x){
      if(L>C.dim(0)/2-1) extend(L);

      TENSOR R({L+1,L+1});
      R(0,0)=1.0;
      float xfact=sqrt(1.0-x*x);
      
      for(int l=1; l<=L; l++){
	R(l,l)=c1(2*l,l)*R(l-1,l-1)*xfact;
	R(l,l-1)=c1(l,l-1)*R(l-1,l-1)*x;
	for(int m=0; m<l-1; m++)
	  R(l,m)=c1(l,m)*R(l-1,m)*x+c2(l,m)*R(l-2,m);
      }

      return R; 
   }


    TENSOR coeffs(const int l, const int dev=0){
      if(dev==0){
	if(l+1>C.dim(0)/2) extend(l);
	return C.rows(2*l,2);
      }
      auto it=remotes.find(dev);
      if(it==remotes.end() || l+1>it->dim(0)/2){
	auto R=Tensor((*this)(l),dev);
	remotes[dev]=R;
	return R.rows(2*l,2);
      }
      return it->rows(2*l,2);
    }


    extend(const int L){
      lock_guard<mutex> lock(mx);
      if(L<=C.dim(0)/2-1) return;
      int _L=C.dim(0)/2-1;
      TENSOR newC({2*(L+1),L+1},0,0);
      newc.block({2*(_L+1),_L+1})=C;
      C.reset(newc);

      for(int l=_L+1; l<=L; l++)
	C(2*l,l)=-sqrt(((float)(2.0*l+1))/(2.0*l-1))*sqrt(1.0/(2.0*l)/(2.0*l-1))*(2.0*l-1);

      for(int l=_L+1; l<=L; l++)
	for(int m=0; m<l; m++){
	  float prefact0=sqrt(((float)(2.0*l+1))/(2.0*l-1));
	  float prefact1=sqrt((float)(l-m))*sqrt(1.0/(l+m));
	  C(2*l,m)=prefact0*prefact1*(2.0*l-1)/(l-m);
	  if(m<l-1){
	    float prefact0b=sqrt(((float)(2.0*l+1))/(2.0*l-3));
	    float prefact2=sqrt((float)(l-m)*(l-m-1))*sqrt(1.0/((l+m)*(l+m-1.0)));
	    C(2*l+1,m)=-prefact0b*prefact2*((float)(l+m-1.0))/(l-m);
	  }
	}
      
    }



  };

}

#endif 
