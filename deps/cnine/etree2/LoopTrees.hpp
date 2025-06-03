#ifndef _LoopTrees
#define _LoopTrees

#include "LoopTree.hpp"
#include "ctree.hpp"


namespace cnine{

  class LoopTrees{
  public:

    typedef ctree_index_set IXSET;

    vector<shared_ptr<LoopTree> > trees;
    map<int,int> contraction_directory;
    vector<vector<vector<int> > > permutation_lists;

    LoopTrees(){}

    LoopTrees(const ctree& ctr){
      for(auto& p:ctr.nodes){
	if(dynamic_pointer_cast<ctree_contraction_node>(p)){
	  auto& x=*dynamic_pointer_cast<ctree_contraction_node>(p);
	  int i=permutation_lists.size();
	  contraction_directory[x.id]=i;
	  permutation_lists.push_back(x.indices.permutations());
	}
      }
      vector<vector<int> > perms;
      add_recursively(ctr,perms);
    }

    void add_recursively(const ctree& ctr, const vector<vector<int> >& perms){
      int n=perms.size();

      if(n==permutation_lists.size()){
	cout<<trees.size()<<endl;
	auto tree=new LoopTree();
	for(int i=ctr.nodes.size()-1; i>=0; i--){
	  auto& x=ctr.nodes[i];
	  if(contraction_directory.find(x->id)!=contraction_directory.end())
	    tree->insert(perms[contraction_directory[x->id]],x);
	  else tree->insert(x);
	}
	trees.push_back(to_share(tree));
	return;
      }

      for(auto& p:permutation_lists[n]){
	auto nperms(perms);
	nperms.push_back(p);
	add_recursively(ctr,nperms);
      }
    }

    void add_recursively(const ctree& ctr, const int j){
      if(j<0) return;
      auto contr=dynamic_pointer_cast<ctree_contraction_node>(ctr.nodes[j]);
      vector<vector<int> > perms;
    }

  };


}

#endif 
