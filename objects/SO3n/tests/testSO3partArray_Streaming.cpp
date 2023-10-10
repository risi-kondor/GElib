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

  int b=200;
  int l=4;
  int nc=32;
  Gdims adims({40,40});

  SO3partArray<float> u=SO3partArray<float>::gaussian(b,adims,l,nc);
  SO3partArray<float> v=SO3partArray<float>::gaussian(b,adims,l,nc);
  //printl("u",u)<<endl;
  //printl("v",v)<<endl;

  //xsSO3partArray<float> w=CGproduct(u,v,2);
  //cout<<w<<endl;
  SO3partArray<float> ws=StreamingCGproduct(u,v,2);
  //cout<<ws<<endl;

  //cout<<DiagCGproduct(u,v,2)<<endl;

  cout<<endl; 
}
