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


#ifndef _CnineContractionTree
#define _CnineContractionTree

#include "TensorView.hpp"
#include "ContractionNode.hpp"


namespace cnine{
  namespace einsum{

  class ContractionTree{
  public:

    shared_ptr<ContractionNode> root;
    set<shared_ptr<ContractionNode> > frontier;
    map<int,set<int> > levels;

    ContractionTree(const EinsumForm& form){
      int i=0;
      for(auto& p:form.args)
	frontier.insert(make_shared<ContractionNode>(i,p,string(static_cast<char>('A'+(i++)),1)));
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    bool levelwise_equal(const ContractionTree& x) const{
      if(levels.size()!=x.levels.size()) return false;
      for(auto& p:levels){
	auto q=x.levels.find(p.first);
	if(q==x.levels.end()) return false;
	if(p.second!=q->second) return false;
      }
      return true;
    }

    int n_ops() const{
      CNINE_ASSRT(root);
      //return root->n_ops();
      return 0;
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    shared_ptr<ContractionNode> add_contraction(const int contraction_id, const string token){
      //cout<<"looking for "<<contraction_id<<endl;
      vector<shared_ptr<ContractionNode> > contr;
      for(auto& p:frontier){
	if(p->contains(contraction_id))
	  contr.push_back(p);
      }
      //cout<<"Found "<<contr.size()<<endl;
      for(auto& p:contr)
	frontier.erase(p);
      auto new_node=make_shared<ContractionNode>(contraction_id,token,contr);
      frontier.insert(new_node);
      levels[new_node->level].insert(contraction_id);
      return new_node;
      //if(frontier.size()==1){
      //root=new_node;
      //}
    }

    
  public: // ---- I/O ----------------------------------------------------------------------------------------


    void latex(ostream& oss) const{
      if(!root) return;
      oss<<"\\[";
      root->latex(oss);
      oss<<"\\]\n";
    }
    
    string str(const string indent="") const{
      ostringstream oss;
      CNINE_ASSRT(root);
      oss<<root->str(indent);
      return oss.str();
    }


    friend ostream& operator<<(ostream& stream, const ContractionTree& x){
      stream<<x.str(); return stream;
    }


  };

  }
}

#endif 
