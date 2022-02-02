#include "GElib_base.cpp"
#include "SO3Fvec.hpp"
#include "GElibSession.hpp"


using namespace cnine;
using namespace GElib;

typedef CtensorB ctensor;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  int b=1;
  int L=2;
  int maxl=2;

  SO3Fvec u=SO3Fvec::gaussian(b,L);
  SO3Fvec v=SO3Fvec::gaussian(b,L);
  printl("u",u)<<endl;
  printl("v",v)<<endl;

  SO3Fvec w=u.Fproduct(v,maxl);
  printl("w",w);
  cout<<endl;

  cout<<"---------------"<<endl; 

  SO3Fvec z=u.Fmodsq(maxl);
  printl("z",z);
  cout<<endl;

}

