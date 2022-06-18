#include "GElib_base.cpp"
#include "SO2_addFFTFn.hpp"
#include "GElibSession.hpp"


using namespace cnine;
using namespace GElib;

typedef CtensorB ctensor;


int main(int argc, char** argv){
  GElibSession session(4);
  cout<<endl;

  int n=10;
  
  CtensorB f=CtensorB::gaussian({10});
  CtensorB F=CtensorB::zero({10});

  SO2_addFFTFn()(unsqueeze0(unsqueeze1(F.view1())),unsqueeze0(unsqueeze1(F.view1())));

  printl("f",f);
  printl("F",F);

  cout<<SO2FmatrixBank.get(5,5)<<endl;

}
