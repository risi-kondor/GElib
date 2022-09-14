#ifndef _SO3CGP_program
#define _SO3CGP_program

#include "GElib_base.hpp"
#include "SO3CGnode.hpp"

namespace GElib{

  //class SO3CGprogramVar; 


  class SO3CGprogram{
  public:

    vector<SO3CGnode*> nodes;

    vector<int> output_nodes;

    SO3CGprogram(){}

    ~SO3CGprogram(){
      for(auto p: nodes) delete p;
      //for(auto p: vars) delete p;
    }


  public:

    SO3CGprogram(const SO3CGprogram& x)=delete;
    SO3CGprogram& operator=(const SO3CGprogram& x)=delete; 


  public:

    int add_node(const int l, const int n){
     int nnodes=nodes.size();
     nodes.push_back(new SO3CGnode(nnodes,l,n));
     return nnodes;
    }

    int add_node(const int l, const int n, SO3CGop* op){
      int nnodes=nodes.size();
      nodes.push_back(new SO3CGnode(nnodes,l,n,op));
      return nnodes;
    }

    int add_input_node(const int l, const int n, const int arg){
      return add_node(l,n,new SO3CG_inputOp(arg));
    }
    


  public:


  };


}


#endif


