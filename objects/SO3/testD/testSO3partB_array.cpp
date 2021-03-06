#include "GElib_base.cpp"
#include "SO3partB_array.hpp"
#include "GElibSession.hpp"


using namespace cnine;
using namespace GElib;

typedef CtensorB ctensor;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  int N=2;
  int B=2;
  Gdims adims({2,2});

  SO3partB_array u=SO3partB_array::gaussian(adims,2,2);
  SO3partB_array v=SO3partB_array::gaussian(adims,2,2);
  printl("u",u)<<endl;
  printl("v",v)<<endl;

  SO3partB_array w=u.CGproduct(v,2);
  cout<<w<<endl;

  cout<<endl; 
}
