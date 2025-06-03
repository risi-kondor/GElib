/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2024, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _GatherMapPack
#define _GatherMapPack

#include "Cnine_base.hpp"
#include "GatherMapB.hpp"
#include "shared_object_pack.hpp"


namespace cnine{

  class GatherMapPack: public shared_object_pack<GatherMapB>{
  public:

    typedef shared_object_pack<GatherMapB> BASE;

    int n_out=0;
    int n_in=0;

    int in_columns=1;
    int out_columns=1;
    int in_columns_n=1;
    int out_columns_n=1;

    vector<int> in_offsets;
    vector<int> out_offsets;

    GatherMapPack(const shared_ptr<GatherMapB>& x){
      BASE::push_back(x);
      n_out=x->n_in;
      n_in=x->n_in;
      in_columns=x->in_columns;
      out_columns=x->out_columns;
      in_columns_n=x->in_columns_n;
      out_columns_n=x->out_columns_n;
      in_offsets.push_back(0);
      out_offsets.push_back(0);
    }

    void push_back(const shared_ptr<GatherMapB>& x){
      BASE::push_back(x);
      CNINE_ASSRT(x->n_in==n_in);
      CNINE_ASSRT(x->in_columns==in_columns);
      CNINE_ASSRT(x->out_columns==out_columns);
      CNINE_ASSRT(x->in_columns_n==in_columns_n);
      CNINE_ASSRT(x->out_columns_n==out_columns_n);
      in_offsets.push_back(in_offsets.back()+x->n_in*in_columns);
      out_offsets.push_back(out_offsets.back()+x->n_out*out_columns);
      n_out+=x->n_out;
    }

    GatherMapPack(const std::vector<shared_ptr<GatherMapB> >& v):
      BASE(v){
      int N=v.size();
      CNINE_ASSRT(N>0);
      in_offsets.resize(N); //reset(Ltensor<int>(Gdims(N)));
      out_offsets.resize(N); //reset(Ltensor<int>(Gdims(N)));
      
      n_in=v[0]->n_in;
      in_columns=v[0]->in_columns;
      out_columns=v[0]->out_columns;
      in_columns_n=v[0]->in_columns_n;
      out_columns_n=v[0]->out_columns_n;

      int in_offs=0;
      int out_offs=0;
      for(int i=0; i<N; i++){
	CNINE_ASSRT(v[i]->n_in==n_in);
	CNINE_ASSRT(v[i]->in_columns==in_columns);
	CNINE_ASSRT(v[i]->out_columns==out_columns);
	CNINE_ASSRT(v[i]->in_columns_n==in_columns_n);
	CNINE_ASSRT(v[i]->out_columns_n==out_columns_n);
	in_offsets[i]=in_offs;
	out_offsets[i]=out_offs;
	in_offs+=v[i]->n_in*in_columns;
	out_offs+=v[i]->n_out*out_columns;
	n_out+=v[i]->n_out;
      }

    }

    int get_nout() const{
      return n_out;
    }

    int get_nin() const{
      return n_in;
    }

    const GatherMapPack& sort() const{
      for(auto& p: *this)
	p->sort();
      return *this;
    }

    string str(const string indent="") const{
      ostringstream oss;
      for(int s=0; s<size(); s++){
	auto& p=(*this)[s];
	for(int i=0; i<p.size(); i++){
	  oss<<indent<<p.target(i)+out_offsets[s]<<"<-(";
	  for(int j=0; j<p.size_of(i); j++){
	    oss<<p(i,j)+in_offsets[s]<<",";
	  }
	  if(p.size_of(i)>0) oss<<"\b";
	  oss<<")\n";
	}
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const GatherMapPack& v){
      stream<<v.str(); return stream;}

    
  };

}

#endif 
