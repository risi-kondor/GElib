#ifndef _it_node
#define _it_node

#include <list>

#include "EinsumForm.hpp"
#include "it_tdef.hpp"
#include "it_instr.hpp"


namespace cnine{

  class it_node{
  public:

    int ix;

    it_node* parent=nullptr;
    list<it_node*> children;

    vector<it_tdef*> tensors;
    vector<it_instr*> instructions; 

    
    it_node(it_node* _parent, const int _ix):
      parent(_parent),
      ix(_ix){
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    virtual string indent(const int depth) const{
      return string(2*depth,' ');
    }

    string code(int depth=0) const{
      ostringstream oss;
      //if(ix==-1) return oss.str(); 
      string istr="i"+to_string(ix);
      int I=3;

      if(ix==-1){
	for(auto& p:tensors)
	  oss<<p->code(depth);
	for(auto& p:children)
	  oss<<p->code(depth);
	return oss.str();
      }

      for(auto& p:tensors)
	oss<<p->code(depth);
      oss<<indent(depth)<<"for(int "<<istr<<"=0; "<<istr<<"<"<<I<<"; ++"<<istr<<"){\n";
      for(auto& p:children)
	oss<<p->code(depth+1);
      for(auto& p:instructions)
	oss<<p->code(depth+1);
      oss<<indent(depth)<<"}\n";
      return oss.str();
    }

  };

}

#endif
