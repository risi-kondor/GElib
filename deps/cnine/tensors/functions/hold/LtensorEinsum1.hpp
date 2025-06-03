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


#ifndef _CnineLtensorEinsum1
#define _CnineLtensorEinsum1

#include "Ltensor.hpp"
#include "EinsumParams.hpp"
#include "EinsumForm1.hpp"


namespace cnine{


  extern void LtensorEinsum1loops(int d, int r , int b, float* R, const float* x, const EsumParams& params);


  class LtensorEinsum1{
  public:

    EinsumForm1 form;

    LtensorEinsum1(const string str):
      form(str){}

    template<typename TYPE>
    Ltensor<TYPE> operator()(const Ltensor<TYPE>& x, vector<int> rdims={}){
      CNINE_ASSRT(rdims.size()==form.bcast_ids.size());
      
      vector<int> dimensions(form.id_tail,-1);
      for(int i=0; i<form.x_ids.size(); i++){
	if(dimensions[form.x_ids[i]]==-1)
	  dimensions[form.x_ids[i]]=x.dims[i];
	else
	  CNINE_ASSRT(dimensions[form.x_ids[i]]==x.dims[i]);
      }

      for(int i=0; i<form.bcast_ids.size(); i++)
	dimensions[form.bcast_ids[i]]=rdims[i];

      auto r_dims=mapcar<int,int>(form.r_ids,[&](const int& id){return dimensions[id];});
      Ltensor<TYPE> R(r_dims,0,x.get_dev());
      add_einsum(R,x,form);
      return R;
    }
      

    template<typename TYPE>
    void add_einsum(const Ltensor<TYPE>& r, const Ltensor<TYPE>& x, const EinsumForm1& form){

      //if(!r.strides.is_decreasing()||!x.strides.is_decreasing())
      //add_einsum(r.with_descreasing_strides(),x.with_decreasing_strides(),
      //  form.permute(r.strides.decreasing_ordering(),x.strides.decreasing_ordering()));

      auto& transfer_indices=form.transfer_indices;
      auto& summation_indices=form.summation_indices;
      auto& broadcast_indices=form.broadcast_indices;

      CNINE_ASSRT(transfer_indices.size()<=4);
      CNINE_ASSRT(summation_indices.size()<=4);
      CNINE_ASSRT(broadcast_indices.size()<=4);

      EsumParams params;
      for(int i=0; i<transfer_indices.size(); i++){
	params.ddims[i]=r.dims[transfer_indices[i].first[0]]; // ???
	params.rstride_d[i]=r.strides.combine(transfer_indices[i].first);
	params.xstride_d[i]=x.strides.combine(transfer_indices[i].second);
      }
      for(int i=0; i<summation_indices.size(); i++){
	params.sdims[i]=x.dims[summation_indices[i][0]];
	params.xstride_s[i]=x.strides.combine(summation_indices[i]);
      }
      for(int i=0; i<broadcast_indices.size(); i++){
	params.bdims[i]=r.dims[broadcast_indices[i][0]];
	params.rstride_b[i]=r.strides.combine(broadcast_indices[i]);
      }

      LtensorEinsum1loops(transfer_indices.size(),summation_indices.size(),broadcast_indices.size(),const_cast<TYPE*>(r.get_arr()), x.get_arr(), params);
    }

   
  };

}

#endif 


    /*
    LtensorEinsum1(const string str){

      auto d1=str.find("->");
      if(d1==string::npos){
	COUT("Error in RtensorEinsumFn: malformed einsum string");
	return;
      }
      auto xstr=str.substr(0,d1);
      auto rstr=str.substr(d1+2,string::npos);
      x_ids=vector<int>(xstr.size());
      r_ids=vector<int>(rstr.size());
      cout<<xstr<<endl;
      cout<<rstr<<endl;

      while(true){
	auto p=rstr.find_first_not_of('x');
	if(p==string::npos) break;
	char c=rstr[p];
	auto v=find_all(rstr,c);
	for(auto q:v) r_ids[q]=id_tail;
	if(xstr.find(c)==string::npos){
	  bstrides.push_back(v);
	  bcast_ids.push_back(id_tail);
	}else{
	  auto w=find_all(xstr,c);
	  for(auto q:w) x_ids[q]=id_tail;
	  dstrides.push_back(make_pair(v,w));
	}
	id_tail++;
      }

      while(true){
	auto p=xstr.find_first_not_of('x');
	if(p==string::npos) break;
	char c=xstr[p];
	auto v=find_all(xstr,c);
	for(auto q:v) x_ids[q]=id_tail;
	sstrides.push_back(v);
	id_tail++;
      }
    }
    */
    /*
    vector<pair<vector<int>,vector<int> > > dstrides;
    vector<vector<int> > sstrides;
    vector<vector<int> > bstrides;
    vector<int> r_ids;
    vector<int> x_ids;
    vector<int> bcast_ids;
    int id_tail=0;
    */
    /*
    void compute(TYPE* r, TYPE* x, const EsumParams& params, const int n_direct, const int n_sum, const int n_bcast){

      switch(n_direct){
      case 0:
	compute_sub(r,x,params,n_sum,n_bcast);
	break;
      case 1:
	for(int i0=0; i0<params.ddims[0]; i0++)
	  compute_sub(r+dstridesr[0]*i0,x+dstridesx[0]*i0,params,n_sum,n_bcast);
	break;
      case 2:
	for(int i0=0; i0<params.ddims[0]; i0++)
	  for(int i1=0; i1<params.ddims[1]; i1++)
	    compute_sub(r+dstridesr[0]*i0+dstridesr[1]*i1,x+dstridesx[0]*i0+dstridesx[1]*i1,params,n_sum,n_bcast);
	break;
      case 3:
	for(int i0=0; i0<params.ddims[0]; i0++)
	  for(int i1=0; i1<params.ddims[1]; i1++)
	    for(int i2=0; i2<params.ddims[2]; i2++)
	      compute_sub(r+dstridesr[0]*i0+dstridesr[1]*i1+dstridesr[2]*i2,
		x+dstridesx[0]*i0+dstridesx[1]*i1+dstridesx[2]*i2,params,n_sum,n_bcast);
	break;
      case 4:
	for(int i0=0; i0<params.ddims[0]; i0++)
	  for(int i1=0; i1<params.ddims[1]; i1++)
	    for(int i2=0; i2<params.ddims[2]; i2++)
	      compute_sub(r+dstridesr[0]*i0+dstridesr[1]*i1+dstridesr[2]*i2+dstridesr[3]*i3,
		x+dstridesx[0]*i0+dstridesx[1]*i1+dstridesx[2]*i2+dstridesx[3]*i3,params,n_sum,n_bcast);
	break;
      }
      }
    */
    
    /*
    inline vector<int> find_all(string& str, const char c) const{
      vector<int> r;
      auto p=str.find_first_of(c);
      while(p!=string::npos){
	str.replace(p,1,1,'x');
	r.push_back(p);
	p=str.find_first_of(c);
      }
      return r;
    }
    */
