#ifndef _contr_node
#define _contr_node

#include "for_node.hpp"

namespace cnine{
  namespace etree{

    class contr_node: public etree_node{
    public:

      int ix;
      shared_ptr<tensor_node> result;
      vector<shared_ptr<tensor_node> > factors;

      contr_node(const int _ix, shared_ptr<etree_node>& _result,  
	const vector<shared_ptr<etree_node> >& _factors):
	ix(_ix),
	result(dynamic_pointer_cast<tensor_node>(_result)){
	for(auto& p: _factors)
	  factors.push_back(dynamic_pointer_cast<tensor_node>(p));
      }


    public: // ---------------------------------------------------------------------------------------------


      virtual void write_to(code_env& env){
	string ixs=to_string(ix);
	env.add_line("float t=0;");
	env.write("for(int i"+ixs+"=0; i"+ixs+"<I"+ixs+"; i"+ixs+"++){");
	env.depth++;
	string factrs;
	for(int i=0; i<factors.size(); i++){
	  auto& x=*factors[i];
	  factrs+="T"+to_string(x.tid)+"("+x.indices.str()+")";
	  if(i<factors.size()-1) factrs+="*";
	}
	env.add_line("t+="+factrs+";");
	env.depth--;
	env.write("}");
	env.add_line("T"+to_string(result->id)+"("+result->indices.str()+")+=t;");
      }
      
    };
  }
}

#endif 
