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


#ifndef __Llists
#define __Llists

#include "Cnine_base.hpp"
#include "Llist.hpp"


namespace cnine{


  template<typename TYPE>
  class Llists{
  public:

    vector<Llist<TYPE> > lists;


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str() const{
      ostringstream oss;
      oss<<"{";
      int i=0;
      for(auto& p:*this){
	oss<<p.first;
	if(i++<size()-1) oss<<",";
      }
      oss<<"}";
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Llist<TYPE>& x){
      stream<<x.str(); return stream;
    }




  };

}


#endif 


