
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _IntegerPartitionObj
#define _IntegerPartitionObj

#include "IntegerPartition.hpp"
#include "YoungTableau.hpp"


namespace GElib{

  class IntegerPartitionObj{
  public:

    int n;
    IntegerPartition lambda;
    vector<IntegerPartitionObj*> parents;

    vector<YoungTableau*> tableaux;

    ~IntegerPartitionObj(){
      for(auto p: tableaux) delete p;
    }


  public: // Constructors 

    IntegerPartitionObj(const IntegerPartition& _lambda): 
      lambda(_lambda), n(_lambda.getn()){
    }


  public: 

    const vector<YoungTableau*>& get_YoungTableaux(){
      if(tableaux.size()==0) make_tableaux();
      return tableaux;
    }


  public: 

    void make_tableaux(){
      //cout<<"Making tableaux for "<<lambda<<endl;
      if(n==1){
	tableaux.push_back(new YoungTableau(1,cnine::fill_identity()));
	return;
      }
      for(auto parent: parents){
	int k=parent->lambda.height();
	int i=k;
	for(int j=0; j<k; j++)
	  if(parent->lambda.p[j]<lambda.p[j]) {i=j; break;}
	const vector<YoungTableau*>& subtableaux=parent->get_YoungTableaux();
	for(auto _a:subtableaux){
	  YoungTableau* t=new YoungTableau(*_a);
	  t->add(i,n);
	  tableaux.push_back(t);
	}
      }
    }

  };

}

#endif 

