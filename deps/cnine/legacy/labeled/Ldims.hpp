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


#ifndef __Ldims
#define __Ldims

#include "Cnine_base.hpp"
#include "GindexSet.hpp"


namespace cnine{


  class Ldims: public vector<int>{
  public:


    Ldims(){}

    Ldims(const vector<int>& x):
      vector<int>(x){
    }


  public: // ---- Copying ------------------------------------------------------------------------------------


    virtual Ldims* clone() const{
      return new Ldims(*this);
    }


  public: // ---- Conversions --------------------------------------------------------------------------------



  public: // ---- Access -------------------------------------------------------------------------------------


    int total() const{
      int t=1; for(int i=0; i<size(); i++) t*=(*this)[i];
      return t;
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    virtual string name() const{
      return "dims";
    }

    virtual string str() const{
      ostringstream oss;
      //oss<<indent;
      oss<<name()<<"(";
      for(int i=0; i<size(); i++){
	oss<<(*this)[i];
	if(i<size()-1) oss<<",";
      }
      oss<<")";
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Ldims& x){
      stream<<x.str(); return stream;}


  };

}

#endif 

