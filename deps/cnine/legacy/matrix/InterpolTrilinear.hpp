/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2021, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _InterpolTrilinear
#define _InterpolTrilinear

#include "Cnine_base.hpp"
#include "RtensorA.hpp"
#include "array_pool.hpp"
#include "CSRmatrix.hpp"


namespace cnine{

  template<typename TYPE>
  class InterpolTrilinear: public CSRmatrix<TYPE>{
  public:

    using CSRmatrix<TYPE>::arr;
    using CSRmatrix<TYPE>::arrg;
    using CSRmatrix<TYPE>::tail;
    using CSRmatrix<TYPE>::memsize;
    using CSRmatrix<TYPE>::dev;
    using CSRmatrix<TYPE>::is_view;
    using CSRmatrix<TYPE>::dir;
    using CSRmatrix<TYPE>::n;

    using CSRmatrix<TYPE>::reserve;
    using CSRmatrix<TYPE>::size;
    using CSRmatrix<TYPE>::offset;
    using CSRmatrix<TYPE>::size_of;
    using CSRmatrix<TYPE>::set_at;


    InterpolTrilinear(const RtensorA& M, const int n0, const int n1, const int n2):
      CSRmatrix<TYPE>(M.get_dim(0),n0*n1){
      CNINE_ASSRT(M.get_dim(1)==3);

      vector<int> len(n,0);
      int total=0;

      for(int i=0; i<n; i++){
	float x0=M(i,0);
	float x1=M(i,1);
	float x2=M(i,2);
	CNINE_ASSRT(x0>=0 && x0<=n0-1);
	CNINE_ASSRT(x1>=0 && x1<=n1-1);
	CNINE_ASSRT(x2>=0 && x2<=n2-1);

	int xb0=std::floor(x0);
	int xb1=std::floor(x1);
	int xb2=std::floor(x2);
	float d0=x0-xb0;
	float d1=x1-xb1;
	float d2=x2-xb2;

	float w000=(1.0-d0)*(1.0-d1)*(1.0-d2);
	float w001=(1.0-d0)*(1.0-d1)*d2;
	float w010=(1.0-d0)*(d1)*(1.0-d2);
	float w011=(1.0-d0)*(d1)*d2;
	float w100=(d0)*(1.0-d1)*(1.0-d2);
	float w101=(d0)*(1.0-d1)*d2;
	float w110=(d0)*(d1)*(1.0-d2);
	float w111=(d0)*(d1)*(d2);

	len[i]+=(w000>0)+(w010>0)+(w100>0)+(w110>0)+(w001>0)+(w011>0)+(w101>0)+(w111>0);
	total+=len[i];
      }

      reserve(2*total);
      tail=0;
      for(int i=0; i<n; i++){
	dir.set(i,0,tail);
	dir.set(i,1,2*len[i]);
	tail+=2*len[i];
      }

      for(int i=0; i<n; i++){
	float x0=M(i,0);
	float x1=M(i,1);
	float x2=M(i,2);
	CNINE_ASSRT(x0>=0 && x0<=n0-1);
	CNINE_ASSRT(x1>=0 && x1<=n1-1);
	CNINE_ASSRT(x2>=0 && x2<=n2-1);

	int xb0=std::floor(x0);
	int xb1=std::floor(x1);
	int xb2=std::floor(x2);
	float d0=x0-xb0;
	float d1=x1-xb1;
	float d2=x2-xb2;

	float w000=(1.0-d0)*(1.0-d1)*(1.0-d2);
	float w001=(1.0-d0)*(1.0-d1)*d2;
	float w010=(1.0-d0)*(d1)*(1.0-d2);
	float w011=(1.0-d0)*(d1)*d2;
	float w100=(d0)*(1.0-d1)*(1.0-d2);
	float w101=(d0)*(1.0-d1)*d2;
	float w110=(d0)*(d1)*(1.0-d2);
	float w111=(d0)*(d1)*(d2);

	int k=0;
	int s0=n0*n1;
	int s1=n1;
	int s2=1;
	float t=w000+w001+w010+w011+w100+w101+w110+w111;
	if(w000>0) set_at(i,k++,xb0*s0+xb1*s1+xb2*s2,w000/t);
	if(w001>0) set_at(i,k++,xb0*s0+xb1*s1+(xb2+1)*s2,w001/t);
	if(w010>0) set_at(i,k++,xb0*s0+(xb1+1)*s1+xb2*s2,w010/t);
	if(w011>0) set_at(i,k++,xb0*s0+(xb1+1)*s1+(xb2+1)*s2,w011/t);
	if(w100>0) set_at(i,k++,(xb0+1)*s0+xb1*s1+xb2*s2,w100/t);
	if(w101>0) set_at(i,k++,(xb0+1)*s0+xb1*s1+(xb2+1)*s2,w101/t);
	if(w110>0) set_at(i,k++,(xb0+1)*s0+(xb1+1)*s1+xb2*s2,w110/t);
	if(w111>0) set_at(i,k++,(xb0+1)*s0+(xb1+1)*s1+(xb2+1)*s2,w111/t);

      }
	
    }

  };


}


#endif   
