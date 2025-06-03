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


#ifndef __Larray
#define __Larray

#include "Cnine_base.hpp"
#include "Ldims.hpp"

namespace cnine{


  class Larray: public Ldims{
  public:

    Larray(){}

    Larray(const vector<int>& x):
      Ldims(x){}


  public: // ---- Copying ------------------------------------------------------------------------------------

    
    virtual Larray* clone() const{
      return new Larray(*this);
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    virtual string name() const{
      return "array";
    }

    string str(const string indent="") const{
      ostringstream oss(indent);
      oss<<"array(";
      for(int i=0; i<size(); i++){
	oss<<(*this)[i];
	if(i<size()-1) oss<<",";
	oss<<")";
      }
      return oss.str();
    }

  };


  // ---- Functions ------------------------------------------------------------------------------------------


  inline Larray operator*(const Larray& x, const Larray& y){
    CNINE_ASSERT(x.size()==1||x.size()==2,"first operand of product must be a vector or a matrix");
    CNINE_ASSERT(x.size()==1||y.size()==2,"second operand of product must be a vector or a matrix");
    if(x.size()==1 && y.size()==2){
      CNINE_ASSRT(x[0]==y[0]);
      return Larray({y[1]});
    }
    if(x.size()==2 && y.size()==1){
      CNINE_ASSRT(x[1]==y[0]);
      return Larray({x[0]});
    }
    if(x.size()==2 && y.size()==2){
      CNINE_ASSRT(x[1]==y[0]);
      return Larray({x[0],y[1]});
    }
    return Larray();
  }

}

#endif 

