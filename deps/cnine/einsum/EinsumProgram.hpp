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


#ifndef _CnineEinsumProgram
#define _CnineEinsumProgram

#include "TensorView.hpp"
#include "einsum_node.hpp"
#include "contraction_node.hpp"


namespace cnine{
  namespace einsum{


  class EinsumProgram{
  public:

    shared_ptr<einsum_node> root;
    set<shared_ptr<einsum_node> > frontier;
    map<int,set<int> > levels;


    EinsumProgram(const vector<shared_ptr<einsum_node> >& _args){
      for(auto& p:_args)
	frontier.insert(p);
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    bool levelwise_equal(const EinsumProgram& x) const{
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


    void add_contraction(int contraction_id){
      cout<<"looking for "<<contraction_id<<endl;
      vector<shared_ptr<einsum_node> > contr;
      for(auto& p:frontier){
	if(p->contains(contraction_id))
	  contr.push_back(p);
      }
      for(auto& p:contr)
	frontier.erase(p);
      auto new_node=make_shared<contraction_node>(contraction_id,contr);
      frontier.insert(new_node);
      levels[new_node->level].insert(contraction_id);
      if(frontier.size()==1){
	root=new_node;//*frontier.begin();
      }
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


    friend ostream& operator<<(ostream& stream, const EinsumProgram& x){
      stream<<x.str(); return stream;
    }


  };

  }
}

#endif 
