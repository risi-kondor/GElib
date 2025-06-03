/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _ComplementSpace
#define _ComplementSpace

//#include "Tensor.hpp"
#include "ColumnSpace.hpp"


namespace cnine{

  template<typename TYPE>
  class ComplementSpace{
  public:

    TensorView<TYPE> M;
    int ncols=0;

    ComplementSpace(const TensorView<TYPE>& _A, TYPE threshold=10e-5){
      //T(M.dims,fill_zero()){
      auto A=ColumnSpace<TYPE>(_A)();

      CNINE_ASSRT(A.ndims()==2);
      const int n=A.dims[0];
      const int m=A.dims[1];

      vector<TYPE> norms(m);
      for(int i=0; i<m; i++){
	norms[i]=A.col(i).norm2();
      }

      M=TensorView<TYPE>({n,n},fill_zero());
      ncols=0;

      for(int i=0; i<n; i++){
	auto col=TensorView<TYPE>({n},fill_zero());
	col.set(i,1);

	// Orthogonalize to columns of A
	for(int j=0; j<m; j++)
	  if(norms[j]>10e-6){
	    col.subtract(A.col(j),inp(col,A.col(j))/norms[j]);
	    //cout<<col.str("Col=")<<endl;
	  }
	//cout<<col.str("cola=")<<endl;

	// Orthogonalize to columns extracted so far
	for(int j=0; j<ncols; j++)
	  col.subtract(M.col(j),inp(col,M.col(j)));
	
	// Normalize    
	TYPE norm=col.norm();
	if(norm>threshold){
	  M.col(ncols).add(col,1.0/norm);
	  ncols++;
	}
 
      }

      //T=M.block({n,ncols},{0,0});
    }
      
    operator TensorView<TYPE>() const{
      return M.block({M.dims[0],ncols});
    }

    TensorView<TYPE> operator()() const{
      return M.block({M.dims[0],ncols});
    }

    TensorView<TYPE> padded() const{
      return M;
    }

    


  };

}

#endif 
