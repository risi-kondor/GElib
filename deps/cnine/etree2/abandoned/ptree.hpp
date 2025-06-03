#ifndef _etree_ptree
#define _etree_ptree

#include "etree_node.hpp"
#include "block_node.hpp"
#include "tensor_node.hpp"


namespace cnine{
  namespace etree{

    class ptree{
    public:

      //shared_ptr<etree_node> root;
      //vector<shared_ptr<etree_node> > nodes;
      shared_ptr<etree_node_dir> nodes;

      vector<shared_ptr<ptree> > child_trees;

      vector<int> inputs;
      int output;

      ptree(){
	nodes=to_share(new etree_node_dir());
	auto x=new block_node();
	x->id=0;
	x->parent=-1;
	x->nodes=nodes;
	nodes->add(x);
      }


    public: // ---------------------------------------------------------------------------------------------


      int add_node(const int parent, etree_node* x){
	int id=nodes->size();
	x->id=id;
	x->parent=parent;
	x->nodes=nodes;
	(*nodes)[parent]->children.push_back(id);
	//nodes.push_back(to_share(x));
	nodes->add(x);
	return id;
      }

      int add_input(const int tensor_id, const einsum::ix_tuple& _indices){
	int r=add_node(0,new tensor_node(tensor_id,_indices));
	inputs.push_back(r);
	return r;
      }

      int add_output(const int tensor_id, const einsum::ix_tuple& _indices){
	int r=add_node(0,new tensor_node(tensor_id,_indices));
	output=r;
	return r;
      }


    public: // ---------------------------------------------------------------------------------------------

      /*
      vector<block_node> branchpoints(const einsum::ix_tuple& indices, 
	const vector<shared_ptr<tensor_node> >& deps){
	for(auto& p:deps)
	  p->mark_path_to_root();
	vector<block_node> R;
	return R;
      }
      */

      void write_to(code_env& env){
	(*nodes)[0]->write_to(env);
      }

    };

  }
}

#endif 
