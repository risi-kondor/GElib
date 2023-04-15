#include "GElib_base.cpp"
#include "SO3partArrayC.hpp"
#include "GElibSession.hpp"
//#include "SO3partC.hpp"


using namespace cnine;
using namespace GElib;

typedef CtensorB ctensor;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  Gdims adims({2});
  int l=2;
  int nc=2;

  SO3partArrayC<float> u=SO3partArrayC<float>::gaussian(adims,l,nc);
  SO3partArrayC<float> v=SO3partArrayC<float>::gaussian(adims,l,nc);
  printl("u",u)<<endl;
  printl("v",v)<<endl;

  SO3partArrayC<float> w=CGproduct(u,v,2);
  cout<<w<<endl;

  cout<<DiagCGproduct(u,v,2)<<endl;

  cout<<u(0)<<endl;

  SO3partC<float> a=SO3partC<float>::sequential(l,3);
  cout<<SO3partArrayC<float>({2,2},a)<<endl;

  cout<<endl; 
}
