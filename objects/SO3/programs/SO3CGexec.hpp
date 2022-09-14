#ifndef _SO3CGexec
#define _SO3CGexec

#include "SO3CGprogram.hpp"
#include "SO3partB.hpp"


namespace GElib{

  class SO3pnode{
  public:

    SO3partB* obj;
    bool view_flag=false;

    ~SO3pnode(){
      if(!view_flag) delete obj;
    }


  public:

    SO3pnode(const cnine::_viewof<SO3partB>& x){
      obj=&x.obj;
      view_flag=true;
    }
    


  };



  class SO3CGexec{
  public:

    const SO3CGprogram& prg;

    vector<SO3pnode*> pnodes;

    ~SO3CGexec(){
      for(auto p:pnodes)
	delete p;
    }


  public:

    SO3CGexec(const SO3CGprogram& _prg): 
      prg(_prg){}


  public:

    SO3partB& operator[](const int i){
      return *pnodes[i]->obj;
    }


  public:

    void operator()(SO3vecB& R, const vector<const SO3vecB*>& args){
      int nargs=args.size();
      int nnodes=prg.nodes.size();
      pnodes.resize(nnodes);

      for(int i=0; i<nnodes; i++){
	const SO3CGnode& node=*prg.nodes[i];

	if(node.is_input()){
	  const SO3CG_inputOp& op=*static_cast<SO3CG_inputOp*>(node.ops[0]);
	  assert(op.arg<nargs);
	  const SO3vecB& arg=*args[op.arg];
	  assert(node.l<arg.parts.size());
	  assert(node.n==arg.parts[node.l]->getn());
	  pnodes[i]=new SO3pnode(cnine::viewof(*arg.parts[node.l]));
	  continue;
	}

	node.forward(this);
	//for(auto op: node.ops){
	//op->forward(this);
	//}

      }
    }


  };

}

#endif
