#ifndef _EtreeArgNode
#define _EtreeArgNode

#include "EtreeNode.hpp"

namespace cnine{

  class EtreeArgNode: public EtreeNode{
  public:

    int tindex;
    //vector<int> tensors;

    EtreeArgNode(const int _tindex, const token_string& _indices):
      EtreeNode(_indices),
      tindex(_tindex){}

    /*
    vector<int> get_ancestor_tensors() const{
      vector<int> R;
      for(auto node=this; node=node->parent; node!=nullptr)
	if(dynamic_cast<EtreeTensorNode*>(node)!=nullptr)
	  R.push_back(dynamic_cast<EtreeTensorNode*>(node)->tindex);
      return R;
      }
    */

  public: // ---- I/O ----------------------------------------------------------------------------------------

    
    string cpu_code(const EtreeParams& params, int depth){
      CNINE_ASSRT(children.size()<=1);
      ostringstream oss;
      return oss.str();
    }


  };

}

#endif 
