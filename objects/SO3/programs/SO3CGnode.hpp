#ifndef _SO3CGnode
#define _SO3CGnode

#include "GElib_base.hpp"

namespace GElib{

  class SO3CGexec;


  class SO3CGop{
  public:

    virtual void forward(SO3CGexec* frame){}
    
  };


  
  class SO3CG_inputOp: public SO3CGop{
  public:

    int arg;

  public:

    SO3CG_inputOp(const int _arg):
      arg(_arg){}

  };



  class SO3CGnode{
  public:

    int id;
    int l;
    int n;

    vector<SO3CGop*> ops;

    ~SO3CGnode(){
      for(auto p:ops) delete p;
    }

  public:

    SO3CGnode(const int _id, const int _l, const int _n): id(_id), l(_l), n(_n){}

    SO3CGnode(const int _id, const int _l, const int _n, SO3CGop* op): 
      id(_id), l(_l), n(_n){
      ops.push_back(op);
    }


  public:

    void add_op(SO3CGop* op){
      ops.push_back(op);
    }

    bool is_input() const{
      if(ops.size()!=1) return false;
      if(dynamic_cast<SO3CG_inputOp*>(ops[0])) return true;
      return false;
    }

    void forward(SO3CGexec* frame) const{
      for(auto p: ops)
	p->forward(frame);
    }

  };


}

#endif 
