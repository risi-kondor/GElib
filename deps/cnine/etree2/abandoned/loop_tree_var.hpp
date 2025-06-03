#ifndef _loop_tree_var
#define _loop_tree_var

#include "code_env.hpp"

namespace cnine{

  class loop_tree_var{
  public:

    int tix;
    vector<int> indices;
    vector<int> dependents;

    string limit_str(){
	ostringstream oss;
	for(int i=0; i<indices.size() ; i++){
	  oss<<"n"<<indices[i];
	  if(i<indices.size()-1) oss<<",";
	}
	return oss.str();
      }

    virtual void write_to(code_env& env){
      env.write("TensorView<float> T"+to_string(tix)+"("+limit_str()+");");
    }

  };

}
#endif 

