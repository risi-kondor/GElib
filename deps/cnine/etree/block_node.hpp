#ifndef _block_node
#define _block_node

#include "etree_node.hpp"

namespace cnine{
  namespace etree{

    class block_node: public etree_node{
    public:

      vector<shared_ptr<etree_node> > children;

      shared_ptr<etree_node> add_child(etree_node* x){
	auto p=to_share(x);
	children.push_back(p);
	p->parent=this;
	return p;
      }
	
      virtual void write_to(code_env& env){
	for(auto& p: children)
	  p->write_to(env);
      }
      
    };
  }
}

#endif 
