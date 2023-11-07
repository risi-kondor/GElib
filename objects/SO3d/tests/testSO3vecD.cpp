#include "GElib_base.cpp"
#include "GElibSession.hpp"
#include "SO3vecD.hpp"
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
  SO3typeD tau({1,1,1});

  SO3vecD<float> u=SO3vecD<float>::gaussian().tau(tau).grid(grid);
  cout<<u<<endl;

  SO3vecD<float> v=SO3vecD<float>::gaussian().batch(b).tau(tau).grid(grid);
  cout<<v<<endl;

}
