#ifndef _etree_ptree
#define _etree_ptree

#include "block_node.hpp"

namespace cnine{
  namespace etree{

    class ptree{
    public:

      shared_ptr<etree_node> root;
      vector<shared_ptr<etree_node> > nodes;

      vector<shared_ptr<ptree> > child_trees;

      vector<shared_ptr<tensor_node> > inputs;
      shared_ptr<tensor_node> output;

      ptree(){
	root=make_shared<block_node>();
	root->id=0;
	nodes.push_back(root);
      }


    public: // ---------------------------------------------------------------------------------------------


      shared_ptr<etree_node> add_input(tensor_node* x){
	auto p=dynamic_pointer_cast<tensor_node>(add_node(root,x));
	inputs.push_back(p);
	return p;
      }

      shared_ptr<etree_node> add_output(tensor_node* x){
	output=dynamic_pointer_cast<tensor_node>(add_node(root,x));
	return output;
      }

      shared_ptr<etree_node> add_node(etree_node* x){
	return add_node(root,x);
      }

      shared_ptr<etree_node> add_node(shared_ptr<etree_node>& node, etree_node* x){
	auto r=node->add_child(x);
	r->id=nodes.size();
	nodes.push_back(r);
	return r;
      }


    public: // ---------------------------------------------------------------------------------------------


      vector<block_node> branchpoints(const einsum::ix_tuple& indices, 
	const vector<shared_ptr<tensor_node> >& deps){
	for(auto& p:deps)
	  p->mark_path_to_root();
	vector<block_node> R;
	return R;
      }

      void write_to(code_env& env){
	root->write_to(env);
      }

    };

  }
}

#endif 
