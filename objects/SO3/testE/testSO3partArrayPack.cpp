#include "GElib_base.cpp"
#include "SO3partB_array.hpp"
#include "GElibSession.hpp"
#include "RtensorA.hpp"
#include "SO3partArrayPack.hpp"

using namespace cnine;
using namespace GElib;

typedef RtensorA rtensor;
typedef CtensorB ctensor;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  int n=4;
  int B=2;
  int b=2;
  Gdims adims({n});

  SO3partArrayPack<float> u=SO3partArrayPack<float>::gaussian(2,adims,2,5);
  cout<<u<<endl;

}
