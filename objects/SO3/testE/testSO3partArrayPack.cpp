#include "GElib_base.cpp"
#include "SO3partB_array.hpp"
#include "GElibSession.hpp"
#include "RtensorA.hpp"
#include "Tensor.hpp"
#include "TensorPack.hpp"
#include "SO3partArrayPack.hpp"

using namespace cnine;
using namespace GElib;

typedef RtensorA rtensor;
typedef CtensorB ctensor;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  int n=4;
  Gdims adims({n});

  //Tensor<float> A=Tensor<float>::sequential({2,3,4,5});
  //cout<<A<<endl;

  //TensorPack<float> u=TensorPack<float>::sequential(2,{4,5,5});
  //cout<<u<<endl;

  SO3partArrayPack<float> u=SO3partArrayPack<float>::sequential(2,adims,2,5);
  cout<<u<<endl;

}
