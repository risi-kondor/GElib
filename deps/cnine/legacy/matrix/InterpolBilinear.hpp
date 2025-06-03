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

#ifndef _InterpolBilinear
#define _InterpolBilinear

#include "Cnine_base.hpp"
#include "RtensorA.hpp"
#include "array_pool.hpp"
#include "CSRmatrix.hpp"


namespace cnine{

  template<typename TYPE>
  class InterpolBilinear: public CSRmatrix<TYPE>{
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


    InterpolBilinear(const RtensorA& M, const int n0, const int n1):
      CSRmatrix<TYPE>(M.get_dim(0),n0*n1){
      
      CNINE_ASSRT(M.get_dim(1)==2);
      //int n=M.get_dim(0);

      vector<int> len(n,0);
      int total=0;

      for(int i=0; i<n; i++){
	float x=M(i,0);
	float y=M(i,1);
	CNINE_ASSRT(x>=0 && x<=n0-1);
	CNINE_ASSRT(y>=0 && y<=n1-1);

	int xb=std::floor(x);
	int yb=std::floor(y);
	float d0=x-xb;
	float d1=y-yb;

	float w00=(1.0-d0)*(1.0-d1);
	float w01=(1.0-d0)*(d1);
	float w10=(d0)*(1.0-d1);
	float w11=(d0)*(d1);

	len[i]+=(w00>0)+(w01>0)+(w10>0)+(w11>0);
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
	CNINE_ASSRT(x0>=0 && x0<=n0-1);
	CNINE_ASSRT(x1>=0 && x1<=n1-1);

	int xb0=std::floor(x0);
	int xb1=std::floor(x1);
	float d0=x0-xb0;
	float d1=x1-xb1;

	float w00=(1.0-d0)*(1.0-d1);
	float w01=(1.0-d0)*(d1);
	float w10=(d0)*(1.0-d1);
	float w11=(d0)*(d1);

	int k=0;
	float t=w00+w01+w10+w11;
	if(w00>0) set_at(i,k++,xb0*n0+xb1,w00/t);
	if(w01>0) set_at(i,k++,xb0*n0+(xb1+1),w01/t);
	if(w10>0) set_at(i,k++,(xb0+1)*n0+xb1,w10/t);
	if(w11>0) set_at(i,k++,(xb0+1)*n0+(xb1+1),w11/t);
      }
	
    }

  };


}


#endif   
