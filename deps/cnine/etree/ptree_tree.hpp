#ifndef _ptree_tree
#define _ptree_tree

#include "ptree.hpp"

namespace cnine{
  namespace etree{

    class ptree_tree{
    public:
      
      shared_ptr<ptree> root;

      ptree_tree(){
	root=make_shared<ptree>();
      }

      shared_ptr<etree_node> add_input(tensor_node* x){


    };

  }
}

#endif
