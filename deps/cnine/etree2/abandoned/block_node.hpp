#ifndef _block_node
#define _block_node

#include "etree_node.hpp"

namespace cnine{
  namespace etree{

    class block_node: public etree_node{
    public:

      virtual string str() const{
	return "block_node";
      }

      virtual void write_to(code_env& env){
	for(auto& p: children)
	  (*nodes)[p]->write_to(env);
      }
      
    };
  }
}

#endif 
