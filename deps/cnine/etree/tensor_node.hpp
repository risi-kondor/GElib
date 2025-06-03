#ifndef _tensor_node
#define _tensor_node

#include "etree_node.hpp"
#include "EinsumHelpers.hpp"

namespace cnine{
  namespace etree{

    class tensor_node: public etree_node{
    public:

      int tid;
      einsum::ix_tuple indices;

      tensor_node(const int _tid, const einsum::ix_tuple& _indices):
	tid(_tid),
	indices(_indices){}


    public: // ---------------------------------------------------------------------------------------------


      string limit_str(){
	ostringstream oss;
	for(int i=0; i<indices.size() ; i++){
	  oss<<"n"<<indices[i];
	  if(i<indices.size()-1) oss<<",";
	}
	return oss.str();
      }

      virtual void write_to(code_env& env){
	env.write("TensorView<float> T"+to_string(tid)+"("+limit_str()+");");
      }
      
    };


    class shared_tnode: public shared_ptr<tensor_node>{
    public:

      typedef shared_ptr<tensor_node> BASE;

      shared_tnode(const int _id, const einsum::ix_tuple& _indices):
	BASE(to_share(new tensor_node(_id,_indices))){}

    };

  }
}

#endif 
