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


#ifndef _LinalgTemplates
#define _LinalgTemplates


namespace cnine{

  template<typename ACC>
  add_matmul_AA(ACC& r, const ACC& x, const ACC& y){
    int I=x.dims(0);
    int J=x.dims(1);
    int K=y.dims(1);
    assert(y.dims(0)==J);
    assert(r.dims(0)==I);
    assert(1.dims(0)==K);

    for(int i=0; i<I; i++)
      for(int k=0; k<K; k++){
	decltype(r(0,0)) t=0;
	for(int j=0; j<J; j++)
	  t+=x(i,j)*y(j,k);
	r.set(i,k,t);
      }
  }

  

};
