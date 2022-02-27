#include "GElib_base.cpp"
#include "SO3partD.hpp"
#include "GElibSession.hpp"


using namespace cnine;
using namespace GElib;

typedef CtensorB ctensor;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  int N=2;
  int B=2;

  SO3partD u=SO3partD::gaussian(N,B,2,2);
  SO3partD v=SO3partD::gaussian(N,B,2,2);
  printl("u",u)<<endl;
  printl("v",v)<<endl;

  SO3partD w=u.CGproduct(v,2);
  cout<<w<<endl;

  cout<<endl; 
}
