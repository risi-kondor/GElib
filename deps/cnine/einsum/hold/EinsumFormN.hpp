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


#ifndef _CnineEinsumFormN
#define _CnineEinsumFormN

#include "TensorView.hpp"
#include "EinsumForm1.hpp"
#include "einsum_node.hpp"
#include "multivec.hpp"

namespace cnine{


  class EinsumFormN: EinsumFormBase{
  public:

    vector<vector<pair<int,int> > > contractions;
    vector<int> contraction_ids;
    shared_ptr<einsum_node> result_node;
    vector<shared_ptr<einsum_node> > arg_nodes;
    int id_tail=0;

    EinsumFormN(const string str){

      //cout<<str<<endl;
      auto d1=str.find("->");
      if(d1==string::npos){
	CNINE_ERROR(str+" is not a well formed einsum string because it has no rhs.");
	return;
      }
      auto lhs=str.substr(0,d1);
      auto rhs=str.substr(d1+2,string::npos);
 
      result_node=make_shared<einsum_node>(rhs.size(),"R");

      vector<string> input_str;
      int offs=0;
      while(offs<lhs.size()){
	auto d=lhs.find(",",offs);
	if(d==string::npos) d=lhs.size();
	input_str.push_back(lhs.substr(offs,d-offs));
	arg_nodes.push_back(make_shared<einsum_node>(d-offs,
	    string(1,static_cast<char>('A'+arg_nodes.size()))));
	offs=d+1;
      }

      while(true){
	auto p=rhs.find_first_not_of('_');
	if(p==string::npos) break;
	char c=rhs[p];
	rhs.replace(p,1,1,'_');
	result_node->ids[p]=id_tail;
	
	for(int j=1; j<input_str.size(); j++){
	  auto q=input_str[j].find(c);
	  if(q==string::npos) continue;
	  input_str[j].replace(q,1,1,'_');
	  arg_nodes[j]->ids[q]=id_tail;
	}
	id_tail++;
      }

      for(int i=0; i<input_str.size(); i++){
	while(true){
	  auto p=input_str[i].find_first_not_of('_');
	  if(p==string::npos) break;
	  vector<pair<int,int> > contr;

	  char c=input_str[i][p];
	  input_str[i].replace(p,1,1,'_');
	  contr.push_back(make_pair(i,(int)p));
	  arg_nodes[i]->ids[p]=id_tail;

	  for(int j=i+i; j<input_str.size(); j++){
	    auto q=input_str[j].find(c);
	    if(q==string::npos) continue;
	    input_str[j].replace(q,1,1,'_');
	    contr.push_back(make_pair(j,(int)q));
	    arg_nodes[j]->ids[q]=id_tail;
	  }

	  if(contr.size()>1){
	    contraction_ids.push_back(id_tail);
	    contractions.push_back(contr);
	  }
	  id_tail++;
	}
      }
      
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      for(auto& p:contractions){
	oss<<"(";
	for(auto& q:p)
	  oss<<q.first<<":"<<q.second<<",";
	oss<<"\b)"<<endl;
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const EinsumFormN& x){
      stream<<x.str(); return stream;
    }


  };

}

#endif 
