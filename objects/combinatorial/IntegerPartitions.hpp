
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _IntegerPartitions
#define _IntegerPartitions

#include "CombinatorialBank.hpp"


namespace GElib{

  class IntegerPartitions{
  public:

    const int n;
    const vector<IntegerPartition*>* lambda;
    bool is_view=false;

    ~IntegerPartitions(){
      if(!is_view){
	for(auto p:*lambda) delete p;
	delete lambda;
      }
    }

  public: // Constructors 

    IntegerPartitions(const int _n): n(_n){
      lambda=&_combibank->get_IntegerPartitions(n);
      is_view=true;
    }


  public: // Access

    int size() const{
      return lambda->size();
    }

    IntegerPartition operator[](const int i){
      return *(*lambda)[i];
    }

  };

}

#endif 
