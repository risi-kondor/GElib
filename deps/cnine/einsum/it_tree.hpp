#ifndef _it_tree
#define _it_tree

#include "ContractionTree.hpp"
#include "it_node.hpp"


namespace cnine{

  class it_tree{
  public:

    it_node* root=nullptr;

    it_tree(const index_set& rindices){
      root=new it_node(nullptr,-1);
      root->tensors.push_back(new it_tdef(0,rindices));
    }

    it_tree(const ContractionTree& x):
      it_tree(x.root->indices){
      get_args(*x.root);
      translate(root,*x.root);
    }


    void get_args(const ContractionNode& x){
      if(x.arg_id>=0)
	root->tensors.push_back(new it_tdef(x.arg_id,x.indices));
      else
	for(auto p:x.children)
	  get_args(*p);
    }


    void translate(it_node* base, const ContractionNode& x){
      if(x.arg_id>=0) return;
      auto parent=find_insertion_point(base,x.indices);
      auto new_node=new it_node(parent,x.contraction_index);
      parent->children.push_back(new_node);
      for(auto p:x.children)
	translate(parent,*p);
    }

    it_node* find_insertion_point(it_node* base, const index_set& indices){

      vector<it_node*> path;
      for(it_node* x=base; x!=root; x=x->parent)
	path.push_back(x);

      vector<int> index_vec;
      for(auto& p:indices)
	index_vec.push_back(p);

      int i=0;
      for(;i<path.size() && i<indices.size(); i++){
	if(path[path.size()-i-1]->ix!=index_vec[i]) break;
      }
      
      it_node* divergence_node=root;
      if(i>0) divergence_node=path[path.size()-i-1]->parent;

      auto node=divergence_node;
      for(int j=i; j<indices.size(); j++){
	auto new_node=new it_node(node,index_vec[j]);
	node->children.push_front(new_node);
	node=new_node;
      }

      return node;
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string code(int depth=0) const{
      ostringstream oss;
      oss<<root->code();
      return oss.str();
    }


  };

}

#endif
