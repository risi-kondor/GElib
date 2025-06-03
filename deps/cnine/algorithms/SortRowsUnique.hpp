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

#ifndef _CnineFindPlantedSubgraphs2
#define _CnineFindPlantedSubgraphs2

#include <set>

#include "Cnine_base.hpp"
#include "TensorView.hpp"
#include "tensor1_view.hpp"


namespace cnine{

  template<typename TYPE>
  class SortRowsUnique{
  public:

    TensorView<TYPE> r;

    SortRowsUnique(const TensorView<TYPE>& x){
      CNINE_ASSRT(x.dev==0);
      CNINE_ASSRT(x.ndims()==2);
      int N=x.dim(0);

      set<tensor1_view<TYPE> > hashmap;
      for(int i=0; i<N; i++)
	hashmap.insert(tensor1_view<TYPE>(x.arr.ptr()+x.strides[0]*i,x.dims[1],x.strides[1]));

      int i=0;
      TensorView<TYPE> R(Gdims({(int)hashmap.size(),x.dim(1)}));
      for(auto& p:hashmap)
	tensor1_view<TYPE>(R.arr.ptr()+R.strides[0]*(i++),R.dims[1],R.strides[1])=p;

      r.reset(R);
    }

    operator TensorView<TYPE>(){
      return r;
    }

  };

}

#endif 

