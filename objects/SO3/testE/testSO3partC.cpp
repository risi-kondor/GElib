#include "GElib_base.cpp"
#include "GElibSession.hpp"
#include "SO3partC.hpp"

using namespace cnine;
using namespace GElib;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  int n=4;
  Gdims adims({n});

  SO3partC<float> u=SO3partC<float>::gaussian(2,5);
  cout<<u<<endl;

}
