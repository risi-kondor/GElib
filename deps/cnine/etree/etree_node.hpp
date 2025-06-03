#ifndef _etree_node
#define _etree_node

#include "code_env.hpp"

namespace cnine{
  namespace etree{

    class etree_node{
    public:

      int id;
      etree_node* parent=nullptr;
      bool flag=false;


    public: // ---------------------------------------------------------------------------------------------


      virtual shared_ptr<etree_node> add_child(etree_node* x){
	CNINE_ERROR("can't add child");
      }

      void mark_path_to_root(){
	flag=true;
	if(parent) parent->mark_path_to_root(); 
      }

      string indent(const int depth){
	return string(2*depth,' ');
      }

      virtual void write_to(code_env& env){
	env.write("");
      }

    };
  }
}

#endif 
