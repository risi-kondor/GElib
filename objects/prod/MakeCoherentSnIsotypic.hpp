// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef _MakeCoherentSnIsotypic
#define _MakeCoherentSnIsotypic

#include "IntegerPartition.hpp"


namespace GElib{


  class MakeCoherentSnIsotypic{
  public:

    typedef cnine::Tensor<double> Tensor;
    typedef Snob2::IntegerPartition IP;


    const IP& lambda;
    const map<IP,SnIsotypicSpace<double> >& sources;
    const Tensor& U;

    int M;
    int N;
    map<IP,int> d_rho;
    map<IP,int> offset;
    set<IP> remaining;
    SnIsotypicSpace<double> result;


    MakeCoherentSnIsotypic(const IP& _lambda, const map<IP,SnIsotypicSpace<double> >& _sources, const Tensor& _U):
      lambda(_lambda), 
      sources(_sources),
      U(_U){
      //result(lambda,Snob2::SnIrrep(lambda).dim(),_sources.begin()->second.dmult(),_sources.begin()->second.dims[2],cnine::fill_zero()){

      GELIB_ASSRT(sources.size()>0);
      M=sources.begin()->second.dmult();
      N=sources.begin()->second.dims[2];

      result=SnIsotypicSpace<double>(lambda,Snob2::SnIrrep(lambda).dim(),M,N,cnine::fill_zero());

      int t=0;
      lambda.for_each_sub([&](const IP& mu){
	  d_rho[mu]=Snob2::SnIrrep(mu).dim();
	  offset[mu]=t;
	  t+=d_rho[mu];
	});

      for(int m=0; m<M; m++){
	lambda.for_each_sub([&](const IP& mu){
	    remaining.insert(mu);});
	add_one_subspace(m,offset.begin()->first,cnine::UnitVec<double>(M,m));
	if(remaining.size()>0) cout<<"Error: not all subrepresentations found."<<endl;
      }
    }

    SnIsotypicSpace<double> operator()() const{
      return result;
    }

    MakeCoherentSnIsotypic(const MakeCoherentSnIsotypic& x)=delete;
	  
    
  private:

    void add_one_subspace(int m, const IP mu, const Tensor& _v){

      //cout<<"Adding "<<mu<<" "<<_v<<endl;
      for(int d=0; d<d_rho[mu]; d++)
	result.slice(1,m).row(offset[mu]+d)=_v*sources.at(mu).slice(0,d);
      remaining.erase(mu);
      if(remaining.size()==0) return;

      auto S=result.slice(1,m)*U;

      map<IP,Tensor> vecs;
      for(auto nu:remaining){
	//cout<<"Trying "<<nu<<endl;
	Tensor P((S*sources.at(nu).matrix().transp()).split1(d_rho[nu],M).fuse01());
	auto v=cnine::sum(0,P);
	if(v.norm()>10e-5){
	  v.normalize();
	  vecs.emplace(nu,v);
	}
      }

      for(auto p:vecs)
	add_one_subspace(m,p.first,p.second);
    }

  };

}

#endif 
