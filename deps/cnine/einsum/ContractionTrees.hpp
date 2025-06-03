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


#ifndef _CnineContractionTrees
#define _CnineContractionTrees

#include "TensorView.hpp"
#include "EinsumForm.hpp"
#include "ContractionTree.hpp"
#include "LatexDoc.hpp"


namespace cnine{
  namespace einsum{


  class ContractionTrees{
  public:

    vector<ContractionTree> trees;

    vector<string> tokens;

    ContractionTrees(const EinsumForm& form):
      tokens(form.tokens){
      ContractionTree T(form);
      build_trees(T,form.contraction_indices);
    }


  public: // ---- Building all possible programs to express form ---------------------------------------------


    void build_trees(const ContractionTree& tree, const vector<int>& rem){
      if(rem.size()==0){
	for(auto& x:trees)
	  if(tree.levelwise_equal(x)) return;
	trees.push_back(tree);
	return;
      }
      for(int i=0; i<rem.size(); i++){
	ContractionTree _tree(tree);
	auto _node=_tree.add_contraction(rem[i],tokens[rem[i]]);
	if(rem.size()==1){
	  _tree.root=_node;
	}
	vector<int> _rem(rem);
	_rem.erase(_rem.begin()+i);
	build_trees(_tree,_rem);
      }
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------

  
    /*
    void latex(string filename="temp") const{
      ostringstream oss;
      for(auto& prg:programs)
	prg.latex(oss);
      LatexDoc doc(oss.str());
      ofstream ofs(filename+".tex");
      ofs<<doc;
      ofs.close();
    }
    */
    
    string str(const string indent="") const{
      ostringstream oss;
      int i=0;
      for(auto& p:trees){
	oss<<indent<<"Tree "<<i++<<":"<<endl;
	oss<<p.str(indent+"  ")<<endl;
      }
      return oss.str();
    }


    friend ostream& operator<<(ostream& stream, const ContractionTrees& x){
      stream<<x.str(); return stream;
    }


  };

  }
}

#endif 
