
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _CombinatorialBank
#define _CombinatorialBank

#include "GElib_base.hpp"
#include "CombinatorialBankLevel.hpp"


namespace GElib{

  class CombinatorialBank{
  public:
    
    vector<CombinatorialBankLevel*> levels;
    
    ~CombinatorialBank(){
      for(auto p:levels) delete p;
    }

  public: 

    const vector<IntegerPartition*>& get_IntegerPartitions(const int n){
      CombinatorialBankLevel& l=level(n);
      return l.get_IntegerPartitions();
    }

    const vector<YoungTableau*>& get_YoungTableaux(const IntegerPartition& lambda){
      CombinatorialBankLevel& l=level(lambda.getn());
      return l.get_YoungTableaux(lambda);
    }


  private:

    CombinatorialBankLevel& level(const int n){
      if(n<=levels.size()) return *levels[n-1];
      const int _n=levels.size();
      levels.resize(n);
      for(int i=_n+1; i<=n; i++){
	if(i==1) levels[i-1]=new CombinatorialBankLevel();
	else levels[i-1]=new CombinatorialBankLevel(i,levels[i-2]);
      }
      return *levels[n-1];
    }

  };
  

}

#endif 
