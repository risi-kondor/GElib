
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _YoungTableaux
#define _YoungTableaux

#include "CombinatorialBank.hpp"


namespace GElib{

  class YoungTableaux{
  public:

    IntegerPartition p;
    const vector<YoungTableau*>* tableaux;
    bool is_view=false;

    ~YoungTableaux(){
      if(!is_view){
	for(auto p:*tableaux) delete p;
	delete tableaux;
      }
    }

  public:

    YoungTableaux(const IntegerPartition& _p): p(_p){
      tableaux=&_combibank->get_YoungTableaux(p);
      is_view=true;
    }


  public: // Access

    int size() const{
      return tableaux->size();
    }

    YoungTableau operator[](const int i){
      return *(*tableaux)[i];
    }

  };

}

#endif 
