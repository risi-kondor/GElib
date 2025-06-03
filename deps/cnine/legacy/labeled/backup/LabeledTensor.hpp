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


#ifndef _CnineLabeledTensor
#define _CnineLabeledTensor

#include "Tensor.hpp"
#include "Ldims.hpp"
#include "LdimsList.hpp"

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
  class LabeledTensor: public Tensor<TYPE>{
  public:

    //vector<Ldims*> ldims;
    LdimsList ldims;

    ~LabeledTensor(){
      //for(auto p:ldims)
      //delete p;
    }


  public:

  public: // ---- Constructors ------------------------------------------------------------------------------


    LabeledTensor(const initializer_list<Ldims>& _ldims, const int _dev=0):
      Tensor<TYPE>(Gdims(convert(_ldims)),_dev),
      ldims(_ldims){}
    //ldims(convertB(_ldims)){}


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      oss<<indent<<"Tensor["<<ldims<<"]:";
      oss<<TensorView<TYPE>::str(indent);
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const LabeledTensor& x){
      stream<<x.str(); return stream;}


  };

}

#endif 
