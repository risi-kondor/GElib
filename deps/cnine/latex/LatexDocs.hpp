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


#ifndef _LatexDocs
#define _LatexDocs

#include "LatexDoc.hpp"

namespace cnine{


  class LatexDocs: public vector<LatexDoc>{
  public:

    void compile(string name){
      for(int i=0; i<size(); i++)
	(*this)[i].compile(name+"_"+to_string(i));
    }

  };

}

#endif 
