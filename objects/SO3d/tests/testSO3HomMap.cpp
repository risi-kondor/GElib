#include "GElib_base.cpp"
#include "GElibSession.hpp"
#include "SO3vecD.hpp"
#include "SO3HomMap.hpp"
#include "Ltensor.hpp"

using namespace cnine;
using namespace GElib;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  int b=3;
  int l=2;
  int n=2;
  Gdims grid(2,2);
  SO3typeD tau1({1,1});
  SO3typeD tau2({2,3});

  SO3HomMap<float> w=SO3HomMap<float>::gaussian(tau1,tau2);
  cout<<w<<endl;

  SO3vecD<float> u=SO3vecD<float>::gaussian().tau(tau1); //.grid(grid);

  auto v=u*w;
  cout<<v<<endl;

}
