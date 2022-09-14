#ifndef _SO3CGP_addCGproductOp
#define _SO3CGP_addCGproductOp

#include "SO3CGexec.hpp"


namespace GElib{

  class SO3CGP_addCGproductOp: public SO3CGop{
  public:

    int ret;
    int arg0;
    int arg1;
    int offs;

  public:

    SO3CGP_addCGproductOp(const int _ret, const int _arg0, const int _arg1, const int _offs):
      ret(_ret), arg0(_arg0), arg1(_arg1), offs(_offs){}

    SO3CGP_addCGproductOp(const SO3CGnode& rnode, const SO3CGnode& xnode, const SO3CGnode& ynode, const int _offs):
      ret(rnode.id), arg0(xnode.id), arg1(ynode.id), offs(_offs){}

  public:

    void forward(SO3CGexec* frame){
      SO3part_addCGproductFn()((*frame)[ret],(*frame)[arg0],(*frame)[arg1],offs);
    }

  };

}

#endif 


