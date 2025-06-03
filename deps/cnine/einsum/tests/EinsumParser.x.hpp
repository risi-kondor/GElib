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


#ifndef _CnineEinsumParser
#define _CnineEinsumParser

#include "TensorView.hpp"
#include "EinsumForm1.hpp"
#include "input_node.hpp"
#include "multivec.hpp"

namespace cnine{


  class ix_entry: public vector<pair<int,int> >{
  public:

    string symbol;
    int type;

    ix_entry(){}

    ix_entry(const string s):
      symbol(s){}

  };



  class EinsumParser{
  public:

    vector<ix_entry> ix_entries;
    vector<vector<vector<int> > > map_to_dims;
    vector<shared_ptr<input_node> > input_nodes;

    EinsumParser(const string str){

      auto d1=str.find("->");
      if(d1==string::npos)
	CNINE_ERROR(str+" is not a well formed einsum string because it has no rhs.");
      auto lhs=str.substr(0,d1);
      auto rhs=str.substr(d1+2,string::npos);

      int offs=0;
      vector<string> arg_str;
      while(offs<lhs.size()){
	auto d=lhs.find(",",offs);
	if(d==string::npos) d=lhs.size();
	arg_str.push_back(lhs.substr(offs,d-offs));
	offs=d+1;
      }

      vector<map<string,int> > tokens;
      auto [d,o]=tokenize(rhs);
      tokens.push_back(d);
      map_to_dims.push_back(o);

      for(auto& p:arg_str){
	auto [d,o]=tokenize(p);
	tokens.push_back(d);
	map_to_dims.push_back(o);
	arg_nodes.push_back(make_shared<einsum_node>(d.size(),
	    string(1,static_cast<char>('A'+arg_nodes.size()))));
      }

      for(int i=0; i<tokens.size(); i++){
	for(auto& p:tokens[i]){
	  ix_entry ix(p.first);
	  ix.push_back(make_pair(i,p.second));
	  for(int j=i+1; j<tokens.size(); j++){
	    auto it=tokens[j].find(p.first);
	    if(it!=tokens[j].end()){
	      ix.push_back(make_pair(j,it->second));
	      tokens[j].erase(it);
	    }
	  }
	  ix_entries.push_back(ix);
	}
      }

    }


  private:


    pair<map<string,int>,vector<vector<int> > > tokenize(const string& str){

      map<string,vector<int> > r;
      int p=0;
      for(int i=0; p<str.size(); i++){
	if(str[p]!='('){
	  r[string({str[p]})].push_back(i);
	  p++;
	}else{
	  auto q=str.find_first_of(')',p+1);
	  if(q==string::npos) 
	    CNINE_ERROR("Unterminated '()'");
	  r[str.substr(p+1,q-p-1)].push_back(i);
	  p=q+1;
	}
      }

      int i=0;
      map<string,int> tokens;
      vector<vector<int> >  occurrences;
      for(auto& p:r){
	tokens[p.first]=i++;
	occurrences.push_back(p.second);
      }

      return make_pair(tokens,occurrences);
    }

  };

}

#endif 


    /*
   pair<string,vector<int> > get_next_token(string& str){
      auto p=str.find_first_not_of('_');
      if(p==string::npos) 
	return  pair<string,vector<int> >("",{});

      string c;
      vector<int>
      if(input_str[p]!='('){
	c=input_str[p];
      }else{
	int q=str.find_first_of(')');

      }
      
    }
    */
      /*
      string c;
      while(true){
	auto p=rhs.find_first_not_of('_');
	if(p==string::npos) break;
	if(input_str[p]!='('){
	  c=input_str[p];
	}else{
	  auto p=rhs.find_first_of('_');
	}

	char c=input_str[i][p];
	input_str[i].replace(p,1,1,'_');
	result_node->ids[p]=id_tail;

      }
      */
