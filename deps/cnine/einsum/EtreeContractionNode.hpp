#ifndef _EtreeContractionNode
#define _EtreeContractionNode

#include "EtreeTensorNode.hpp"

namespace cnine{

  class EtreeContractionNode: public EtreeNode{
  public:

    //int index;

    EtreeContractionNode(const int _ix){
      index=_ix;
    }

    EtreeContractionNode(const int _ix, const index_set& _external_indices):
      EtreeNode(_external_indices){
      index=_ix;
    }

  public: // ---- I/O ----------------------------------------------------------------------------------------

    
    string cpu_code(const EtreeParams& params, int depth){
      CNINE_ASSRT(children.size()<=1);
      ostringstream oss;
      string istr="i"+to_string(index);
      int I=params.dim(index);
      oss<<indent(depth)<<"float t=0;\n";
      oss<<indent(depth)<<"for(int "<<istr<<"=0; "<<istr<<"<"<<I<<"; ++"<<istr<<"){\n";
      oss<<indent(depth+1)<<"t+=0;"<<endl;
      oss<<indent(depth)<<"}"<<endl;
      return oss.str();
    }

    virtual string str(const string indent="") const{
      ostringstream oss;
      oss<<indent<<"contract "<<index<<":"<<endl;
      for(auto& p:children)
	oss<<p->str(indent+"  ")<<endl;
      for(auto& p:subtrees)
	oss<<p->str(indent+"  ")<<endl;
      return oss.str();
    }


  };

}
#endif 
