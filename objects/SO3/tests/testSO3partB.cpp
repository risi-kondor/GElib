#include "GElib_base.cpp"
#include "SO3partB.hpp"
#include "GElibSession.hpp"


using namespace cnine;
using namespace GElib;

typedef CtensorB ctensor;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;


  SO3partB u=SO3partB::gaussian(1,2,2);
  SO3partB v=SO3partB::gaussian(1,2,2);
  printl("u",u)<<endl;
  printl("v",v)<<endl;

  SO3partB w=u.CGproduct(v,2);
  cout<<123<<endl;
  cout<<w<<endl;

  cout<<endl; 
}
