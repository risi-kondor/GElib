#ifndef _LoopTreeOps
#define _LoopTreeOps

#include "code_env.hpp"
#include "ctree_index_set.hpp"
#include "TikzTreeNode.hpp"


namespace cnine{


  class LoopTree_tensor_registry: public unordered_map<int,ctree_index_set >{
  public:
    
  };


  class LoopTree_contr{
  public:

    int id;
    int ix;
    vector<int> args;
    vector<int> dependents;

    LoopTree_contr(const int _id, const int _ix, const vector<int>& _args, const vector<int>& _dependents): 
      id(_id),
      ix(_ix),
      args(_args),
      dependents(_dependents){}
    
  };


  class LoopTreeTensorNode{
  public:

    int id;
    //int ix;
    ctree_index_set indices;
    
    LoopTreeTensorNode(const int _id, const ctree_index_set& _indices):
      id(_id),
      indices(_indices){}

    shared_ptr<LoopTreeTensorNode> shareable_copy() const{
      return make_shared<LoopTreeTensorNode>(id,indices);
    }


  public: // ---- OUTPUT ------------------------------------------------------------------------------------


    void write_to(code_env& env){
      env.write("TensorView<float> T"+to_string(id)+"("+indices.limit_str()+");");
    }

    string str() const{
      return string("TensorView<float> T"+to_string(id)+"("+indices.limit_str()+");");
    }

  };


  class LoopTreeContractionNode{
  public:

    shared_ptr<LoopTree_tensor_registry> registry;

    int id;
    int ix;
    vector<int> args;
    
    
    LoopTreeContractionNode(const shared_ptr<LoopTree_tensor_registry> _registry,
      const int _id, const int _ix, const vector<int> _args):
      registry(_registry),
      id(_id),
      ix(_ix),
      args(_args){}


  public: // ---- OUTPUT ------------------------------------------------------------------------------------


    void write_to(code_env& env){
      string ixs=to_string(ix);
      env.add_line("float t=0;");
      env.write("for(int i"+ixs+"=0; i"+ixs+"<I"+ixs+"; i"+ixs+"++){");
      env.depth++;
      string factrs;
      for(int i=0; i<args.size(); i++){
	factrs+="T"+to_string(args[i])+"("+(*registry)[args[i]].index_str()+")";
	if(i<args.size()-1) factrs+="*";
      }
      env.add_line("t+="+factrs+"");
      env.depth--;
      env.write("}");
      env.add_line("T"+to_string(id)+"("+(*registry)[id].index_str()+")+=t;");
    }

    void to_tikz(TikzTreeNode& parent) const{
      string rlabel("\\parbox{4cm}{\\texttt{");
      rlabel+="float t=0\\\\ \n";
      rlabel+="for(int i"+to_string(ix)+"=0; i"+to_string(ix)+"<I"+to_string(ix)+"; i"+to_string(ix)+"++)\\\\ \n";
      string factrs;
      for(int i=0; i<args.size(); i++){
	factrs+="T"+to_string(args[i])+"("+(*registry)[args[i]].index_str()+")";
	if(i<args.size()-1) factrs+="*";
      }
      rlabel+="\\phantom{MM}t+="+factrs+"\\\\";
      rlabel+="}}";
      auto x=parent.add_child(to_string(ix),rlabel);
    }
  };

}
#endif 
