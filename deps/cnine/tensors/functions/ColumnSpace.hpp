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

#ifndef _ColumnSpace
#define _ColumnSpace

#include "TensorView.hpp"


namespace cnine{

  template<typename TYPE>
  class ColumnSpace{
  public:

    TensorView<TYPE> T;
    int ncols=0;

    ColumnSpace(const TensorView<TYPE>& M, TYPE threshold=10e-5):
      T(M.get_dims(),fill_zero()){
      CNINE_ASSRT(M.ndims()==2);
      const int m=M.dim(1);

      for(int i=0; i<m; i++){
	TensorView<TYPE> col=const_cast<TensorView<TYPE>&>(M).col(i);

	for(int j=0; j<ncols; j++)
	  col.subtract(T.col(j),inp(col,T.col(j)));
      
	TYPE norm=col.norm();
	//cout<<"norm="<<norm<<endl;
	if(norm>threshold){
	  T.col(ncols).add(col,1.0/norm);
	  ncols++;
	}
      }
      //cout<<"cols="<<ncols<<endl;
    }

    operator TensorView<TYPE>(){
      return T.block({T.dim(0),ncols});
    }

    //operator Ltensor<TYPE>(){
    //return T.block({T.dim(0),ncols});
    //}

    TensorView<TYPE> operator()(){
      return T.block({T.dims[0],ncols});
    }



  };

}

#endif 
