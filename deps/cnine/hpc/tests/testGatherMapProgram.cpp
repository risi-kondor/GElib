#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "GatherMapProgram.hpp"
#include "GatherRows.hpp"
#include "Ltensor.hpp"

using namespace cnine;

int main(int argc, char** argv){

  cnine_session session;

  int n=10;

  GatherMapB* dummy=new GatherMapB();

  GatherMapProgram prog(dims(10,1),dims(10,1));

  GatherMapVar in(prog.input());
  GatherMapVar v1(prog,dims(3,4));
  GatherMapVar v2(prog,dims(2,2));

  prog.gather(v1,in(1,0),dummy);

  cout<<prog<<endl;

}

