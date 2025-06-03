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


#ifndef _CnineLtensorView
#define _CnineLtensorView

#include "Cnine_base.hpp"
#include "TensorView.hpp"

#ifdef _WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif 

#ifdef _WITH_CUBLAS
#include <cublas_v2.h>
extern cublasHandle_t cnine_cublas;
#endif 


namespace cnine{

  template<typename TYPE>
  class LtensorView: public TensorView<TYPE>{
  public:

    LdimsList ldims;

    ~LtensorView(){
    }


  public: // ---- Constructors ------------------------------------------------------------------------------


    LtensorView(const MemArr<TYPE>& _arr, const LdimsList& _ldims, const GstridesB& _strides):
      TensorView<TYPE>(_arr,Gdims(_ldims),_strides),
      ldims(_ldims){}



  public: // ---- Copying -----------------------------------------------------------------------------------


    LtensorView(const LtensorView<TYPE>& x):
      TensorView<TYPE>(x), 
      ldims(x.ldims){}

    LtensorView& operator=(const LtensorView& x){
      assert(ldims==x.ldims);
      TensorView<TYPE>::operator=(x);
    }


  public: // ---- Conversions -------------------------------------------------------------------------------


    LtensorView(const TensorView<TYPE>& x, const LdimsList& _ldims):
      TensorView<TYPE>(x),
      ldims(_ldims){}

    LtensorView(TensorView<TYPE>&& x, const LdimsList& _ldims):
      TensorView<TYPE>(std::move(x)),
      ldims(_ldims){}


  public: // ---- Devices -----------------------------------------------------------------------------------

    

  public: // ---- Batches -----------------------------------------------------------------------------------

    
    bool is_batched() const{
      return ldims.is_batched();
    }

    int nbatch() const{
      return ldims.nbatch();
    }

    LtensorView batch(const int b) const{
      if(!is_batched()) return *this;
      return LtensorView(TensorView<TYPE>::slice(0,b),ldims.remove(0));
    }

    void batchwise(std::function<void(int, const LtensorView&)> lambda) const{
      if(!is_batched()){
	lambda(0,*this);
	return;
      }
      int B=nbatch();
      for(int b=0; b<B; b++){
	lambda(b,batch(b));
      }
    }

    static LdimsList common_batch(const LtensorView& x, const LtensorView& y){
      /*
      if(x.is_batched()){
	if(y.is_batched()){
	  CNINE_ASSRT(x.nbatch()==y.nbatch());
	  return LdimsList({Lbatch(x.nbatch())});
	}else{
	  return LdimsList({Lbatch(x.nbatch())});
	}
      }else{
	if(y.is_batched()){
	  return LdimsList({Lbatch(x.nbatch())});
	}else{
	  return LdimsList();
	}
      }
      */
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      oss<<indent<<"LtensorView["<<ldims<<"]:"<<endl;
      if(!is_batched())
	oss<<TensorView<TYPE>::str(indent+"  ");
      else{
	batchwise([&](const int b, const LtensorView& x){
	    oss<<"Batch "<<b<<":"<<endl;
	    oss<<x.str(indent)<<endl;
	  });
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const LtensorView& x){
      stream<<x.str(); return stream;}


  };

}

#endif 
    /*
    const LtensorView batch(const int b) const{
      if(!is_batched()) return *this;
      return LtensorView(const_cast<LtensorView&>(*this).TensorView<TYPE>::slice(0,b),ldims.remove(0));
    }
    */

    /*
    void batchwise(std::function<void(int, const LtensorView&)> lambda) const{
      if(!is_batched()){
	lambda(0,*this);
	return;
      }
      int B=nbatch();
      for(int b=0; b<B; b++)
	lambda(b,batch(b));
    }
    */

    /*
    static vector<vector<int> > convert(const initializer_list<Ldims>& _ldims){
      vector<vector<int> > R;
      for(auto& p:_ldims)
	R.push_back(p);
      return R;
    }
    */
