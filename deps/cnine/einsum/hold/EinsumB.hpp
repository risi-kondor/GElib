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


#ifndef _CnineEinsumB
#define _CnineEinsumB

#include "TensorView.hpp"
#include "EinsumFormB.hpp"
#include "EinsumPrograms.hpp"

namespace cnine{


  template<typename TYPE>
  class EinsumB{
  public:

    EinsumFormB form;
    
    shared_ptr<EinsumPrograms> programs;

    TensorView<TYPE> r;
    vector<TensorView<TYPE> > args;

    EinsumB(const string str):
      form(str){
      //programs=make_shared<EinsumPrograms>(form);
    }
    
    template<typename... Args> 
    void add_einsum(const TensorView<TYPE>& _r, const Args&... _args){
      r.reset(_r);
      args.clear();
      unroller(_args...);
      apply();
    }
     
    template<typename... Args> 
    void unroller(const TensorView<TYPE>& x, const Args&... _args){
      args.push_back(x);
      unroller(_args...);
    }

    void unroller(){}
    

  public: // ------------------------------------------------------------------------------------------------


    void apply(){

    }

  };

}

#endif 
