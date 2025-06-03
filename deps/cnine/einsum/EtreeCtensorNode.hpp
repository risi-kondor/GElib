#ifndef _EtreeTensorNode
#define _EtreeTensorNode

#include "EtreeTensorNode.hpp"
#include "EtreeParams.hpp"
#include "EinsumForm.hpp"


namespace cnine{

  class EtreeCtensorNode: public EtreeTensorNode{
  public:

    int cindex;

    EtreeTensorNode(const int _cindex, const int _tindex, const token_string& _indices):
      EtreeTensorNode(_tindex,_indices),
      cindex(_cindex){}
    

  public: // ---- I/O ----------------------------------------------------------------------------------------


    string cpu_code(const EtreeParams& params, int depth){
      ostringstream oss;
      string istr="i"+to_string(tindex);
      oss<<indent(depth)<<"TensorView<float> T"<<tindex<<"(dims("<<params.dims_str(indices)<<"),0,0);"<<endl;
      oss<<indent(depth)<<"float* arr"<<tindex<<"=T"<<tindex<<".get_arr();"<<endl;
      return oss.str();
    }


  };

}

#endif 
