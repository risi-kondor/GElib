#ifndef _EtreeNode
#define _EtreeNode

#include "EinsumForm.hpp"
#include "EtreeParams.hpp"


namespace cnine{

  class EtreeNode{
  public:

    //EtreeNode* parent=nullptr;
    int index=0;
    int rtensor=0;
    token_string indices;
    index_set external_indices;
    index_set internal_indices;

    vector<shared_ptr<EtreeNode> > subtrees;
    vector<shared_ptr<EtreeNode> > children;

    int level=0;
    string name="M";

    EtreeNode(){}
    
    EtreeNode(const token_string& _indices):
      indices(_indices){}
    
    EtreeNode(const index_set& _external_indices):
      external_indices(_external_indices){}
    

  public: // ---- Access -------------------------------------------------------------------------------------


    shared_ptr<EtreeNode> subtree(const int ix){
      for(auto& p:subtrees)
	if(p->index==ix) return p;
      //if(dynamic_cast<EtreeLoopNode&>(*p).index==ix) return p;
      return nullptr;
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    virtual string cpu_code(const EtreeParams& params, int depth)=0;

    virtual string indent(const int depth){
      return string(2*depth,' ');
    }

    virtual string str(const string indent="") const{
      ostringstream oss;
      oss<<indent<<"Unknown"<<endl;
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const EtreeNode& x){
      stream<<x.str(); return stream;
    }

  };

}

#endif 
