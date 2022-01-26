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

  SO3Fvec u=SO3Fvec::gaussian(b,L);
  SO3Fvec v=SO3Fvec::gaussian(b,L);
  printl("u",u)<<endl;
  printl("v",v)<<endl;

  SO3vecB w=u.CGproduct(v,2);
  cout<<w<<endl;

  cout<<endl; 
}

