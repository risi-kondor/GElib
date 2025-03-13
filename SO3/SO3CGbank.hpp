// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2024, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3CGbank
#define _SO3CGbank

#include "Gpart.hpp"
#include "SO3group.hpp"
#include "SO3type.hpp"
#include "SO3CGindex.hpp"

namespace GElib{

  class SO3CGbank{
  public:

    typedef cnine::TensorView<float> FTENSOR;

    unordered_map<SO3CGindex,FTENSOR> coeffsf;
    unordered_map<SO3CGindex,FTENSOR> coeffsf_gpu;

    template<typename TYPE>
    cnine::TensorView<TYPE>& get(const int l1, const int l2, const int l, const int dev=0){
      SO3CGindex ix(l1,l2,l);
      if constexpr(std::is_same<TYPE,float>::value){
	if(dev==0){
	  auto it=coeffsf.find(ix);
	  if(it!=coeffsf.end()) return it->second;
	  coeffsf.emplace(ix,CGmatrix(l1,l2,l));
	  return coeffsf[ix];
	}
	if(dev==1){
	  auto it=coeffsf_gpu.find(ix);
	  if(it!=coeffsf_gpu.end()) return it->second;
	  coeffsf_gpu.emplace(ix,FTENSOR(get<float>(l1,l2,l,0),dev));
	  return coeffsf_gpu[ix];
	}
	GELIB_ERROR("CG matrix must be on CPU or GPU0.");
	return *(new cnine::TensorView<TYPE>());
      }
      GELIB_ERROR("Currenly only single precision CG matrices supported.");
      return *(new cnine::TensorView<TYPE>());
    }

  private:
    
    FTENSOR CGmatrix(const int l1, const int l2, const int l){
      FTENSOR R({2*l1+1,2*l2+1});
      for(int m1=-l1; m1<=l1; m1++)
	for(int m2=std::max(-l2,-l-m1); m2<=std::min(l2,l-m1); m2++)
	  R(m1+l1,m2+l2)=slowCG(l1,l2,l,m1,m2);
      return R;
    }

    double logfact(double n){
      return lgamma(n+1);
    }
    
    double plusminus(int k){ if(k%2==1) return -1; else return +1; }
    
    double slowCG(const int l1, const int l2, const int l, const int m1, const int m2){
      
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
