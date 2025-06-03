#ifndef _EtreeTensorNode
#define _EtreeTensorNode

#include "EtreeNode.hpp"
#include "EtreeParams.hpp"
#include "EinsumForm.hpp"


namespace cnine{

  class EtreeTensorNode: public EtreeNode{
  public:

    int tindex;

    EtreeTensorNode(const int _tindex, const token_string& _indices):
      EtreeNode(_indices),
      tindex(_tindex){}
    

  public: // ---- I/O ----------------------------------------------------------------------------------------


    string cpu_code(const EtreeParams& params, int depth){
      ostringstream oss;
      string istr="i"+to_string(tindex);
      oss<<indent(depth)<<"TensorView<float> T"<<tindex<<"(dims("<<params.dims_str(indices)<<"),0,0);"<<endl;
      oss<<indent(depth)<<"float* arr"<<tindex<<"=T"<<tindex<<".get_arr();"<<endl;
      for(auto& p:children)
	oss<<p->cpu_code(params,depth);
      return oss.str();
    }


  };

}

#endif 
