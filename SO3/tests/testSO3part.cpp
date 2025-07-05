#include "GElib_base.cpp"
//#include "GElibSession.hpp"

#include "SO3part.hpp"
#include "Gfunctions.hpp"

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

  #ifdef _WITH_CUDA
  SO3part<float> xg(x,1);
  SO3part<float> yg(y,1);
  print(xg);
  #endif

  //SO3part<float> x1(irrep=1,batch=2,grid=Gdims({2,2}),filltype=4,channels=3);
  SO3part<float> x1(grid=Gdims({2,2}),batch=2,irrep=1,filltype=4,channels=3);
  print(x1);


  cout<<"CG product:"<<endl;
  auto z=CGproduct(x,y,1);
  print(z);
  #ifdef _WITH_CUDA
  print(CGproduct(xg,yg,1));
  #endif

  auto w=DiagCGproduct(x,y,1);
  print(w);
  #ifdef _WITH_CUDA
  print(DiagCGproduct(xg,yg,1));
  #endif

  //TensorView<float> M({1,3,2},4,0);
  //auto r=SO3part<float>::spharm(M,2);
  //print(r);

}
