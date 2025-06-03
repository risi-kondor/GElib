#ifndef _etree_node
#define _etree_node

#include "code_env.hpp"

namespace cnine{
  namespace etree{

    class etree_node_dir;


    class etree_node{
    public:

      int id;
      int parent=-1;
      vector<int> children;
      shared_ptr<etree_node_dir> nodes;
      //bool flag=false;


    public: // ---------------------------------------------------------------------------------------------


      //virtual void add_child(const int x){
      //CNINE_ERROR("can't add child");
      //}

      //void mark_path_to_root(){
      //flag=true;
      //if(parent) parent->mark_path_to_root(); 
      //}

      string indent(const int depth){
	return string(2*depth,' ');
      }

      virtual string str() const{
	return "etree_node";
      }

      virtual void write_to(code_env& env){
	env.write("");
      }

    };


    class etree_node_dir{
    public:

      vector<shared_ptr<etree_node> > nodes;

      int size(){
	return nodes.size();
      }

      void add(etree_node* x){
	nodes.push_back(to_share(x));
      }

      shared_ptr<etree_node> operator[](const int i){
	CNINE_ASSRT(i<nodes.size());
	return nodes[i];
      }

      etree_node& operator()(const int i){
	CNINE_ASSRT(i<nodes.size());
	return *nodes[i];
      }

    };

  }
}

#endif 
