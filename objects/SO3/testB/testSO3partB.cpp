#include "GElib_base.cpp"
#include "SO3partB.hpp"
#include "GElibSession.hpp"


using namespace cnine;
using namespace GElib;

typedef CtensorB ctensor;


int main(int argc, char** argv){
  GElibSession session(4);
  cout<<endl;

  //long long a=9;
  short a;
  cout<<sizeof(a)<<endl;


  const int B=2;

  SO3partB u=SO3partB::gaussian(B,3,2);
  SO3partB v=SO3partB::gaussian(B,2,2);
  printl("u",u)<<endl;
  printl("v",v)<<endl;

  SO3partB w=u.CGproduct(v,2);
  cout<<w<<endl;

  cout<<endl; 
}
