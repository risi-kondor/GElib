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


#ifndef _FakeGrad
#define _FakeGrad

namespace cnine{

  template<typename OBJ> 
  class FakeGrad{
  public:

    OBJ* grad=nullptr;

    //~FakeGrad(){
    //if(!is_view) delete grad;
    //}


  public:

    void add_to_grad(const OBJ& x){
      if(grad) grad->add(x);
      else grad=new OBJ(x);
    }

    OBJ& get_grad(){
      if(!grad) grad=new OBJ(OBJ::zeros_like(static_cast<OBJ&>(*this)));
      return *grad;
    }

    OBJ view_of_grad(){
      if(!grad) grad=new OBJ(OBJ::zeros_like(static_cast<OBJ&>(*this)));
      return grad->view();
    }

  };

}

#endif 
