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


#ifndef _CnineLtensor
#define _CnineLtensor

#include "Tensor.hpp"
#include "Ldims.hpp"
#include "LdimsList.hpp"
#include "LtensorView.hpp"

#ifdef _WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif 

#ifdef _WITH_CUBLAS
#include <cublas_v2.h>
extern cublasHandle_t cnine_cublas;
#endif 


namespace cnine{

  inline vector<vector<int> > convert(const initializer_list<Ldims>& _ldims){
    vector<vector<int> > R;
    for(auto& p:_ldims)
      R.push_back(p);
    return R;
  }

  /*
  inline vector<Ldims*> convertB(const initializer_list<Ldims>& _ldims){
    vector<Ldims* > R;
    for(auto& p:_ldims)
      R.push_back(p.clone());
    return R;
  }
  */

  template<typename TYPE>
  class Ltensor: public LtensorView<TYPE>{
  public:

    using LtensorView<TYPE>::LtensorView;
    using LtensorView<TYPE>::arr;
    using LtensorView<TYPE>::dims;
    using LtensorView<TYPE>::strides;
    using LtensorView<TYPE>::ldims;
    using LtensorView<TYPE>::is_batched;
    using LtensorView<TYPE>::batchwise;


  public: // ---- Constructors ------------------------------------------------------------------------------


    Ltensor(const LdimsList& _ldims, const int _dev=0):
      LtensorView<TYPE>(MemArr<TYPE>(_ldims.total(),_dev),_ldims,GstridesB(Gdims(_ldims))){}

    Ltensor(const LdimsList& _ldims, const fill_zero& dummy, const int _dev=0): 
      LtensorView<TYPE>(MemArr<TYPE>(_ldims.total(),dummy,_dev),_ldims,GstridesB(Gdims(_ldims))){}

    Ltensor(const LdimsList& _ldims, const fill_sequential& dummy, const int _dev=0):
      Ltensor(_ldims,_dev){
      int N=ldims.total();
      for(int i=0; i<N; i++)
	arr[i]=i;
    }
    

  public: // ---- Named constructors ------------------------------------------------------------------------


    static Ltensor<TYPE> zero(const LdimsList& _ldims, const int _dev=0){
      return Ltensor<TYPE>(_ldims,fill_zero(),_dev);
    }

    static Ltensor<TYPE> sequential(const LdimsList& _ldims, const int _dev=0){
      return Ltensor<TYPE>(_ldims,fill_sequential(),_dev);
    }


  public: // ---- Batches -----------------------------------------------------------------------------------


  public: // ---- I/O ---------------------------------------------------------------------------------------

    
    string str(const string indent="") const{
      ostringstream oss;
      oss<<indent<<"Tensor["<<ldims<<"]:"<<endl;
      if(!is_batched())
	oss<<TensorView<TYPE>::str(indent);
      else{
	batchwise([&](const int b, const LtensorView<TYPE>& x){
	    oss<<"  Batch "<<b<<":"<<endl;
	    oss<<x.str(indent+"  ")<<endl;
	  });
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Ltensor& x){
      stream<<x.str(); return stream;}


  };


  // ---- Functions ------------------------------------------------------------------------------------------


  template<typename TYPE>
  Tensor<TYPE> operator*(const TensorView<TYPE>& x, const TensorView<TYPE>& y){
    
  }

}

#endif 
    /*
    static vector<vector<int> > convert(const initializer_list<Ldims>& _ldims){
      vector<vector<int> > R;
      for(auto& p:_ldims)
	R.push_back(p);
      return R;
    }
    */

