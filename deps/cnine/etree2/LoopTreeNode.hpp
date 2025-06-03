#ifndef _LoopTreeNode
#define _LoopTreeNode

#include "code_env.hpp"
#include "ctree_index_set.hpp"
#include "LoopTreeOps.hpp"
//#include "TikzStream.hpp"
#include "TikzTreeNode.hpp"

namespace cnine{


  class LoopTreeNode{
  public:

    int ix;
    shared_ptr<LoopTree_tensor_registry> registry;

    //vector<shared_ptr<LoopTreeTensorNode> > tensors;
    vector<shared_ptr<LoopTreeTensorNode> > pre_tensors;
    vector<shared_ptr<LoopTreeContractionNode> > ops;
    vector<shared_ptr<LoopTreeNode> > children;

    vector<int> indices;
    vector<int> ops_below;


    LoopTreeNode(const shared_ptr<LoopTree_tensor_registry> _registry):
      registry(_registry),
      ix(-1){}

    LoopTreeNode(const shared_ptr<LoopTree_tensor_registry> _registry, 
      const int _ix, const vector<int>& _indices):
      registry(_registry),
      ix(_ix),
      indices(_indices){
      indices.push_back(ix);
    }

    shared_ptr<LoopTreeNode> recursive_copy_to_share(const shared_ptr<LoopTree_tensor_registry> _registry){
      auto R=new LoopTreeNode(_registry,ix,indices);
      for(auto& p:pre_tensors)
	R.push_back(p->copy_to_share());
      for(auto& p:ops)
	R.push_back(p->copy_to_share());
      for(auto& p:ops)
	R.push_back(p->recursive_copy_to_share(_registry));
    }


  public: // ------------------------------------------------------------------------------------------------


    bool is_parallelizable() const{
      return true;
    }


  public: // ------------------------------------------------------------------------------------------------


    void insert_tensor(int _id, const ctree_index_set& _indices){
      auto new_tensor=new LoopTreeTensorNode(_id,_indices);
      //tensors.push_back(to_share(new_tensor));
      pre_tensors.insert(pre_tensors.begin(),to_share(new_tensor));
      (*registry)[_id]=_indices;
    }


    void insert(vector<int> remaining, const LoopTree_contr& x){
      ops_below.push_back(x.id);
      
      if(remaining.size()==0){
	auto r=new LoopTreeContractionNode(registry,x.id,x.ix,x.args);
	ops.push_back(to_share(r));
      }

      int xix=remaining[0];
      auto it=children.begin();
      while(it!=children.end()){
	auto& child=**it;
	if(child.ix==xix){
	  child.insert(vector<int>(remaining.begin()+1,remaining.end()),x);
	  return;
	}
	if([&](){
	    for(auto& p:x.dependents)
	      //if(child.ops_below.find(p)!=child.ops_below.end()) return true;
	      if(std::find(child.ops_below.begin(),child.ops_below.end(),p)!=child.ops_below.end()) return true;
	    return false;
	  }())
	  break;
	it++;
      }

      auto new_tensor=new LoopTreeTensorNode(x.id,remaining); //x.indices.minus(indices));
      //tensors.push_back(to_share(new_tensor));
      (*registry)[x.id]=remaining;

      auto new_node=new LoopTreeNode(registry,xix,indices);
      new_node->ops_below.push_back(x.id);
      children.insert(it,to_share(new_node));

      new_node->pre_tensors.push_back(to_share(new_tensor));

      for(int j=1; j<remaining.size(); j++){
	auto nnew=new LoopTreeNode(registry,remaining[j],new_node->indices);
	nnew->ops_below.push_back(x.id);
	new_node->children.push_back(to_share(nnew));
	new_node=nnew;
      }
      new_node->ops.push_back(to_share(new LoopTreeContractionNode(registry,x.id,x.ix,x.args)));
	//new_node->insert(vector<int>(remaining.begin()+1,remaining.end()),x);
    }


  public: // ---- OUTPUT ------------------------------------------------------------------------------------


    void write_to(code_env& env){

      //if(pre_tensors.size()>0)
      //env.write("");

      for(auto& p: pre_tensors)
	p->write_to(env);
      
      if(ix>=0){
       	string ixs=to_string(ix);
	env.write("for(int i"+ixs+"=0; i"+ixs+"<I"+ixs+"; i"+ixs+"++){");
	env.depth++;
      }

      for(auto& p: ops)
	p->write_to(env);
      
      //for(auto& p: tensors)
      //p->write_to(env);
      
      for(auto& p: children){
	if(p->children.size()>0) 
	  env.write(""); 
	p->write_to(env);
      }
      if(children.size()>0) 
	env.write(""); 

      if(ix>=0){
	env.depth--;
	env.write("}");
      }
    }

    void to_tikz(TikzTreeNode& parent) const{

      string rlabel("\\parbox{4cm}{\\texttt{");
      for(auto& p: pre_tensors)
	rlabel+=p->str()+"\\\\";
      if(ix>=0)
	rlabel+="for(int i"+to_string(ix)+"=0; i"+to_string(ix)+"<I"+to_string(ix)+"; i"+to_string(ix)+"++)\n";
      rlabel+="}}";
      auto x=parent.add_child(to_string(ix),rlabel);

      for(auto& p: ops)
	p->to_tikz(x);

      for(auto& p: children)
	p->to_tikz(x);
    }

  };

}

#endif 

    //friend TikzStream& operator<<(TikzStream& stream, const LoopTreeNode& v){
    //stream<<"444"; 
    //return stream;
    //}

