#ifndef _etree_ctensor_node
#define _etree_ctensor_node

#include "tensor_node.hpp"
#include "code_blocks.hpp"

namespace cnine{
  namespace etree{

    class ctensor_node: public tensor_node{
    public:

      int cindex;
      vector<shared_ptr<tensor_node> > factors;
      
      ctensor_node(const int _id, const int _cindex, const einsum::ix_tuple& _indices):
	tensor_node(_id,_indices),
	cindex(_cindex){}

      ctensor_node(const int _id, const int _cindex, const einsum::ix_tuple& _indices, 
	const vector<shared_ptr<tensor_node> >& _factors):
	tensor_node(_id,_indices),
	cindex(_cindex),
	factors(_factors){}


    public: // ---- OUTPUT -----------------------------------------------------------------------------------


      void cpu_code(code_env& env){
	env.add_line("TensorView<float> T"+to_string(id)+"("+limit_str()+");");
	for_blocks loops(env,indices);
	env.add_line("float t=0;");
	if(true){
	  for_block loop(env,cindex);
	  string factrs;
	  for(int i=0; i<factors.size(); i++){
	    auto& x=*factors[i];
	    factrs+="T"+to_string(x.id)+"("+x.indices.str()+")";
	    if(i<factors.size()-1) factrs+="*";
	  }
	  env.add_line("t+="+factrs+";");
	}
	env.add_line("T"+to_string(id)+"("+indices.str()+")+=t;");
      }

    };


   class shared_cnode: public shared_ptr<ctensor_node>{
    public:

      typedef shared_ptr<ctensor_node> BASE;

      shared_cnode(const int _id, const int _cindex, const einsum::ix_tuple& _indices):
	BASE(to_share(new ctensor_node(_id,_cindex,_indices))){}

     shared_cnode(const int _id, const int _cindex, const einsum::ix_tuple& _indices,
       const vector<shared_ptr<tensor_node> >& _factors):
       BASE(to_share(new ctensor_node(_id,_cindex,_indices,_factors))){}

    };



  }
}

#endif 
