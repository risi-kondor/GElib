/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2022, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _bidirectional_dag
#define _bidirectional_dag

#include "Cnine_base.hpp"


namespace cnine{

  template<typename BASE>
  class biDAGnode: public BASE{
  public:

    vector<biDAGnode*> parents;
    vector<biDAGnode*> children;

    biDAGnode(const vector<biDAGnode*> _parents):
      parents(_parents){
      for(auto p: parents)
	p->children.push_back(this);
    }

    

  };

}

#endif 
