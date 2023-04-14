#include "GElib_base.cpp"
#include "SO3partArrayC.hpp"
#include "SO3partC.hpp"
#include "GElibSession.hpp"


using namespace cnine;
using namespace GElib;

typedef CtensorB ctensor;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  int N=2;
  int B=2;
  int b=2;
  Gdims adims({2});

  SO3partArrayC<float> u=SO3partArrayC<float>::gaussian(adims,2,2);
  SO3partArrayC<float> v=SO3partArrayC<float>::gaussian(adims,2,2);
  printl("u",u)<<endl;
  printl("v",v)<<endl;

  SO3partArrayC<float> w=CGproduct(u,v,2);
  cout<<w<<endl;

  cout<<u(0)<<endl;

  SO3partC<float> a=SO3partC<float>::sequential(2,3);
  cout<<SO3partArrayC<float>({2,2},a)<<endl;

  cout<<endl; 
}
