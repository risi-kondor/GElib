#ifndef _EtreeLoopNode
#define _EtreeLoopNode

#include "EtreeTensorNode.hpp"

namespace cnine{

  class EtreeLoopNode: public EtreeNode{
  public:

    //int index;
    //vector<int> tensors;

    EtreeLoopNode(const int _ix){
      index=_ix;}
    //index(_ix){}

    /*
    vector<int> get_ancestor_tensors() const{
      vector<int> R;
      for(auto node=this; node=node->parent; node!=nullptr)
	if(dynamic_cast<EtreeTensorNode*>(node)!=nullptr)
	  R.push_back(dynamic_cast<EtreeTensorNode*>(node)->tindex);
      return R;
      }
    */

    shared_ptr<EtreeNode> subtree(const int ix){
      for(auto& p:subtrees)
	if(dynamic_cast<EtreeLoopNode&>(*p).index==ix) return p;
      return nullptr;
    }

  public: // ---- I/O ----------------------------------------------------------------------------------------

    
    string cpu_code(const EtreeParams& params, int depth){
      CNINE_ASSRT(children.size()<=1);
      ostringstream oss;
      string istr="i"+to_string(index);
      int I=params.dimensions[index];
      oss<<indent(depth)<<"for(int "<<istr<<"=0; "<<istr<<"<"<<I<<"; ++"<<istr<<"){"<<endl;;
      oss<<indent(depth+1)<<"float* arr"<<rtensor<<"=arr"<<rtensor<<"+"<<istr<<"*"<<
	params.strides(rtensor,index)<<";\n";

      //for(auto& p:tensors)
      //oss<<indent(depth+1)<<"offset"<<p<<"+="<<istr<<params.strides[p][params.indices[p].find(index)]<<";";
      for(auto& p:children)
	oss<<p->cpu_code(params,depth+1);
      oss<<string(2*depth,' ')<<"}"<<endl;
      return oss.str();
    }

    virtual string str(const string indent="") const{
      ostringstream oss;
      if(index>=0)
	oss<<indent<<"for i"<<index<<"=0 to "<<endl;
      for(auto& p:children)
	oss<<p->str(indent+"  ")<<endl;
      for(auto& p:subtrees)
	oss<<p->str(indent+"  ")<<endl;
      return oss.str();
    }



  };

}

#endif 
