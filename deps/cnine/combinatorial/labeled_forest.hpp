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

#ifndef _labeled_forest
#define _labeled_forest

#include "Cnine_base.hpp"
#include "labeled_tree.hpp"


namespace cnine{

  template<typename TYPE>
  class labeled_forest: public vector<labeled_tree<TYPE>*>{
  public:

    ~labeled_forest(){
      for(auto p:*this)
	delete p;
    }

  public:

    bool contains_rooted_path_consisting_of(const std::vector<TYPE>& x) const{
      set<TYPE> s(x.begin(),x.end());
      return contains_rooted_path_consisting_of(s);
    }

    bool contains_rooted_path_consisting_of(const std::set<TYPE>& x) const{
      for(auto p:*this){
	if(p->contains_rooted_path_consisting_of(x)) return true;
      }
      return false;
    }

  };

}

#endif 
