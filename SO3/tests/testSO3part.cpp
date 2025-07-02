#include "GElib_base.cpp"
//#include "GElibSession.hpp"

#include "SO3part.hpp"
#include "SO3functions.hpp"

using namespace cnine;
using namespace GElib;


int main(int argc, char** argv){
  //GElibSession session;
  cout<<endl;

  int b=3;
  int l=2;
  int n=2;

  SO3part<float> x(1,2,3,4);
  SO3part<float> y(1,2,3,4);
  print(x);

  //SO3part<float> x1(irrep=1,batch=2,grid=Gdims({2,2}),filltype=4,channels=3);
  SO3part<float> x1(grid=Gdims({2,2}),batch=2,irrep=1,filltype=4,channels=3);
  print(x1);


  auto z=CGproduct(x,y,1);
  print(z);

  auto w=DiagCGproduct(x,y,1);
  print(w);

  //TensorView<float> M({1,3,2},4,0);
  //auto r=SO3part<float>::spharm(M,2);
  //print(r);

}
