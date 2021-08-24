
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _CombinatorialBankLevel
#define _CombinatorialBankLevel

#include <unordered_map>

#include "IntegerPartition.hpp"
#include "IntegerPartitionObj.hpp"


namespace GElib{

  class CombinatorialBankLevel{
  public:

    const int n;
    CombinatorialBankLevel* sub=nullptr;

    vector<IntegerPartitionObj*> ip;
    unordered_map<IntegerPartition,IntegerPartitionObj*> ip_map;

    vector<IntegerPartition*> partitions;

    ~CombinatorialBankLevel(){
      for(auto p: partitions) delete p;
      for(auto p: ip) delete p;
    }


  public:

    CombinatorialBankLevel(): n(1){
    }

    CombinatorialBankLevel(const int _n, CombinatorialBankLevel* _sub): 
      n(_n), sub(_sub){
    }


  public: // Access

    const vector<IntegerPartition*>& get_IntegerPartitions(){
      if(partitions.size()==0) make_partitions();
      return partitions;
    }

    const vector<YoungTableau*>& get_YoungTableaux(const IntegerPartition& lambda){
      IntegerPartitionObj& obj=get_IntegerPartitionObj(lambda);
      return obj.get_YoungTableaux();
    }


  private:

    IntegerPartitionObj& get_IntegerPartitionObj(const IntegerPartition& lambda){
      assert(lambda.getn()==n);
      auto it=ip_map.find(lambda);
      if(it!=ip_map.end()) return *it->second;

      IntegerPartitionObj* r=new IntegerPartitionObj(lambda);
      if(n>1){
	//cout<<string(8-n,' ')<<lambda<<endl;
	for(int i=0; i<lambda.k; i++)
	  if(lambda.shortenable(i)){
	    IntegerPartition mu(lambda);
	    mu.remove(i);
	    r->parents.push_back(&sub->get_IntegerPartitionObj(mu));
	  }
      }

      ip.push_back(r);
      ip_map[lambda]=r;
      return *r;
    }

    void make_partitions(){
      if(n==1){
	partitions.push_back(new IntegerPartition(1,cnine::fill_identity()));
	return;
      }
      const vector<IntegerPartition*>& sub_partitions=sub->get_IntegerPartitions();
      for(auto _p: sub_partitions){
	IntegerPartition p=*_p;
	int k=p.height();
	if(k==1||p(k-1)>p(k)){
	  p.add(k);
	  partitions.push_back(new IntegerPartition(p));
	  p.remove(k);
	}
	p.add(k+1);
	partitions.push_back(new IntegerPartition(p));
      }
    }

  };

}

#endif 

