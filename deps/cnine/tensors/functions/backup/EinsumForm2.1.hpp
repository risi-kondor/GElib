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


#ifndef _CnineEinsumForm2
#define _CnineEinsumForm2

#include "TensorView.hpp"
#include "EinsumForm1.hpp"


namespace cnine{


  class EinsumForm2: EinsumFormBase{
  public:

    //vector<bool> convolution_flag;
    vector<vector<int> > x_summation_indices;
    vector<vector<int> > y_summation_indices;
    vector<vector<int> > r_summation_indices;
    vector<pair<vector<int>,vector<int> > > xr_indices;
    vector<pair<vector<int>,vector<int> > > yr_indices;
    vector<pair<vector<int>,vector<int> > > xy_indices;
    vector<vector<vector<int> > > transfer_indices;
    vector<vector<vector<int> > > convolution_indices;
    vector<vector<vector<int> > > triple_contraction_indices;

    vector<int> x_ids;
    vector<int> y_ids;
    vector<int> r_ids;
    int id_tail=0;


    EinsumForm2(const string str){

      auto d0=str.find(",");
      auto d1=str.find("->");
      if(d0==string::npos || d1==string::npos || d0>d1){
	CNINE_ERROR(str+" is not a well formed einsum string.");
	return;
      }
      auto xstr=str.substr(0,d0);
      auto ystr=str.substr(d0+1,d1-d0-1);
      auto rstr=str.substr(d1+2,string::npos);
      x_ids=vector<int>(xstr.size());
      y_ids=vector<int>(ystr.size());
      r_ids=vector<int>(rstr.size());

      while(true){
	
	// find a new index i appearing in r 
	auto p=rstr.find_first_not_of('_');
	if(p==string::npos) break;
	char c=rstr[p];
	auto v=find_all(rstr,c);
	for(auto q:v) r_ids[q]=id_tail;
	bool is_in_x=(xstr.find(c)!=string::npos);
	bool is_in_y=(ystr.find(c)!=string::npos);

	if(is_in_x){
	  auto xw=find_all(xstr,c);
	  for(auto q:xw) x_ids[q]=id_tail;

	  if(is_in_y){
	    auto yw=find_all(ystr,c);
	    if(c=='*'){ // triple contraction
	      triple_contraction_indices.push_back({xw,yw,v});
	    }else{ // transfer or convolution
	      if(c=='U'||c=='V'||c=='W'){
		id_tail++;
		convolution_indices.push_back({xw,yw,v});
		//convolution_flag.push_back(c=='U'||c=='V'||c=='W');
	      }else{
		transfer_indices.push_back({xw,yw,v});
	      }
	    }
	    for(auto q:yw) y_ids[q]=id_tail;
	  }else{ // xr transfer index
	    xr_indices.push_back(make_pair(xw,v));
	  }	  
	}else{
	  if(is_in_y){ // yr transfer index 
	    auto yw=find_all(ystr,c);
	    for(auto q:yw) y_ids[q]=id_tail;
	    yr_indices.push_back(make_pair(yw,v));
	  }else{ // broadcast index 
	    r_summation_indices.push_back(v);
	    r_ids.push_back(id_tail);
	  }
	}
	id_tail++;
      }

      while(true){

	// find a new index i appearing in x
	auto p=xstr.find_first_not_of('_');
	if(p==string::npos) break;
	char c=xstr[p];
	auto v=find_all(xstr,c);
	for(auto q:v) x_ids[q]=id_tail;

	// if i is a summation index 
	if(ystr.find(c)==string::npos)
	  x_summation_indices.push_back(v);
	else{
	  auto w=find_all(ystr,c);
	  for(auto q:w) y_ids[q]=id_tail;
	  xy_indices.push_back(make_pair(v,w));
	}

	id_tail++;
      }

      while(true){
	
	// the remainder are y summation indices
	auto p=ystr.find_first_not_of('_');
	if(p==string::npos) break;
	char c=ystr[p];
	auto v=find_all(ystr,c);
	for(auto q:v) y_ids[q]=id_tail;
	y_summation_indices.push_back(v);
	id_tail++;
      }

    }


    // ---- I/O -----------------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;

      oss<<indent<<"x summations: ";
      for(auto& p:x_summation_indices) oss<<p<<",";
      oss<<"\b"<<endl;

      oss<<indent<<"y summations: ";
      for(auto& p:y_summation_indices) oss<<p<<",";
      oss<<"\b"<<endl;

      oss<<indent<<"xr transfers: ";
      for(auto& p:xr_indices)
	oss<<p.first<<"->"<<p.second<<",";
      oss<<"\b"<<endl;

      oss<<indent<<"yr_transfers: ";
      for(auto& p:yr_indices)
	oss<<p.first<<"->"<<p.second<<",";
      oss<<"\b"<<endl;

      oss<<indent<<"xt contractions: ";
      for(auto& p:xy_indices)
	oss<<p.first<<"*"<<p.second<<",";
      oss<<"\b"<<endl;

      oss<<indent<<"Three way contractions: ";
      for(auto& p:triple_contraction_indices)
	oss<<p[0]<<"*"<<p[1]<<"->"<<p[2]<<",";
      oss<<"\b"<<endl;

      oss<<indent<<"Convolutions: ";
      for(auto& p:convolution_indices)
	oss<<p[0]<<"*"<<p[1]<<"->"<<p[2]<<",";
      oss<<"\b"<<endl;

      oss<<indent<<"Three way transfers: ";
      for(auto& p:transfer_indices)
	oss<<p[0]<<"*"<<p[1]<<"->"<<p[2]<<",";
      oss<<"\b"<<endl;

      oss<<indent<<"Broadcasting: ";
      for(auto& p:r_summation_indices) oss<<p<<",";
      oss<<"\b"<<endl;

      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const EinsumForm2& x){
      stream<<x.str(); return stream;
    }

  };

}

#endif 
