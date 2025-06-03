#ifndef _for_node
#define _for_node

#include "block_node.hpp"

namespace cnine{
  namespace etree{

    class for_node: public block_node{
    public:

      int ix;

      for_node(const int _ix):
	ix(_ix){}

    public: // ---------------------------------------------------------------------------------------------


      virtual void write_to(code_env& env){
	string ixs=to_string(ix);
	env.write("for(int i"+ixs+"=0; i"+ixs+"<I"+ixs+"; i"+ixs+"++){");
	env.depth++;
	block_node::write_to(env);
	//for(auto& p: children)
	//p->write_to(env);
	env.depth--;
	env.write("}");
      }
      
    };
  }
}

#endif 
