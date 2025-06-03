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


#ifndef _CnineEinsumForm1
#define _CnineEinsumForm1

#include "Ltensor.hpp"


namespace cnine{

  class EinsumFormBase{
  public:

  protected:

    inline vector<int> find_all(string& str, const char c) const{
      vector<int> r;
      auto p=str.find_first_of(c);
      while(p!=string::npos){
	str.replace(p,1,1,'_');
	r.push_back(p);
	p=str.find_first_of(c);
      }
      return r;
    }

  };


  class EinsumForm1: EinsumFormBase{
  public:

    vector<pair<vector<int>,vector<int> > > transfer_indices;
    vector<vector<int> > summation_indices;
    vector<vector<int> > broadcast_indices;
    vector<int> r_ids;
    vector<int> x_ids;
    vector<int> bcast_ids;
    int id_tail=0;


    EinsumForm1();

    EinsumForm1(const string str){

      auto d1=str.find("->");
      if(d1==string::npos){
	CNINE_ERROR(str+" is not a well formed einsum string.");
	return;
      }
      auto xstr=str.substr(0,d1);
      auto rstr=str.substr(d1+2,string::npos);
      x_ids=vector<int>(xstr.size());
      r_ids=vector<int>(rstr.size());

      while(true){
	
	// find a new index i appearing in the result
	auto p=rstr.find_first_not_of('_');
	if(p==string::npos) break;
	char c=rstr[p];
	auto v=find_all(rstr,c);
	for(auto q:v) r_ids[q]=id_tail;

	// if i is a broadcast index
	if(xstr.find(c)==string::npos){
	  broadcast_indices.push_back(v);
	  bcast_ids.push_back(id_tail);
	}

	// if i is a transfer index
	else{
	  auto w=find_all(xstr,c);
	  for(auto q:w) x_ids[q]=id_tail;
	  transfer_indices.push_back(make_pair(v,w));
	}

	id_tail++;
      }

      // the remaining indices in the input are summation indices
      while(true){
	auto p=xstr.find_first_not_of('_');
	if(p==string::npos) break;
	char c=xstr[p];
	auto v=find_all(xstr,c);
	for(auto q:v) x_ids[q]=id_tail;
	summation_indices.push_back(v);
	id_tail++;
      }

    }

    EinsumForm1 permute(const vector<int>& r_pi, const vector<int>& x_pi){
      EinsumForm1 R;
      for(auto& p: transfer_indices)
	R.transfer_indices.push_back(make_pair(cnine::permute(p.first,r_pi),cnine::permute(p.second,x_pi)));
      for(auto& p: summation_indices)
	R.summation_indices.push_back(cnine::permute(p,x_pi));
      for(auto& p: broadcast_indices)
	R.broadcast_indices.push_back(cnine::permute(p,r_pi));
      R.x_ids=cnine::permute(x_ids,x_pi);
      R.r_ids=cnine::permute(r_ids,r_pi);
      R.bcast_ids=bcast_ids; // this is doesn't matter because permutation is applied after constructing result
      R.id_tail=id_tail;
      return R;
    }


    // ---- I/O -----------------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;

      oss<<indent<<"Summations: ";
      for(auto& p:summation_indices) oss<<p<<",";
      oss<<"\b"<<endl;

      oss<<indent<<"Transfers: ";
      for(auto& p:transfer_indices)
	oss<<p.second<<"->"<<p.first<<",";
      oss<<"\b"<<endl;

      oss<<indent<<"Broadcasting: ";
      for(auto& p:broadcast_indices) oss<<p<<",";
      oss<<"\b"<<endl;

      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const EinsumForm1& x){
      stream<<x.str(); return stream;
    }

  };

}

#endif 
