#include "GElib_base.cpp"
#include "SO3vecB.hpp"
#include "GElibSession.hpp"


using namespace cnine;
using namespace GElib;

typedef CtensorB ctensor;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  int b=1;
  SO3type tau({2,2,2});

  SO3vecB u=SO3vecB::gaussian(1,tau);
  SO3vecB v=SO3vecB::gaussian(1,tau);
  printl("u",u)<<endl;
  printl("v",v)<<endl;

  SO3vecB w=u.CGproduct(v,2);
  cout<<w<<endl;

  cout<<endl; 
}

