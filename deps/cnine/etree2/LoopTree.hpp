#ifndef _LoopTree
#define _LoopTree

#include "ctree.hpp"
#include "LoopTreeNode.hpp"
#include "TikzTree.hpp"

namespace cnine{


  class LoopTree{
  public:

    typedef ctree_index_set IXSET;

    shared_ptr<LoopTreeNode> root;
    shared_ptr<LoopTree_tensor_registry> registry;

    LoopTree(){
      registry=to_share(new LoopTree_tensor_registry());
      root=to_share(new LoopTreeNode(registry));
    }

    LoopTree(const ctree& _ctr):
      LoopTree(){
      for(int i=_ctr.nodes.size()-1; i>=0; i--)
	insert(_ctr.nodes[i]);
    }


  public: // ------------------------------------------------------------------------------------------------


    void insert(const ctree_tensor_handle& x){
      insert(x.obj);
    }

    void insert(const shared_ptr<ctree_tensor_node>& x){
      if(dynamic_pointer_cast<ctree_contraction_node>(x))
	insert_contraction(*dynamic_pointer_cast<ctree_contraction_node>(x));
      else
	root->insert_tensor(x->id,x->indices);
    }

    void insert_contraction(const ctree_contraction_node& x){
      vector<int> args;
      for(auto& p: x.args)
	args.push_back(p->id);
      LoopTree_contr lcontr(x.id,x.ix,args,x.dependents);
      root->insert(x.indices,lcontr);
    }


    void insert(vector<int> indices, const shared_ptr<ctree_tensor_node>& _x){
      if(dynamic_pointer_cast<ctree_contraction_node>(_x)){
	auto& x=*dynamic_pointer_cast<ctree_contraction_node>(_x);
	vector<int> args;
	for(auto& p: x.args)
	  args.push_back(p->id);
	LoopTree_contr lcontr(x.id,x.ix,args,x.dependents);
	root->insert(indices,lcontr);
      }
      else
	root->insert_tensor(_x->id,_x->indices);
    }



  public: // ------------------------------------------------------------------------------------------------


    void write_to(code_env& env){
      if(root)
	root->write_to(env);
    }

    /*
    string tikz() const{
      TikzStream tstream;
      tstream<<"\\begin{tikzpicture}[";
      tstream<<"every node/.style={circle, draw=black}";
      tstream<<",sibling distance=3cm";
      tstream<<"]\n";
      tstream.oss<<"\\";
      if(root) tstream<<*root;
      tstream.oss<<";\n";
      tstream.write("\\end{tikzpicture}");
      return tstream.str();
    }
    */

    TikzTree tikz_tree() const{
      TikzTree ttree;
      auto x=ttree.add_root("");
      for(auto& p:root->children)
	p->to_tikz(x);
      return ttree;
    }

  };

}

#endif 
