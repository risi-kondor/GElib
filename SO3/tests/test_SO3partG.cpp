#include "GElib_base.cpp"
//#include "GElibSession.hpp"

#include "SO3part.hpp"
#include "CGproduct.hpp"
#include "DiagCGproduct.hpp"

using namespace cnine;
using namespace GElib;


int main(int argc, char** argv){
  //GElibSession session;
  #ifdef _WITH_CUDA
  cout<<endl;

  int b=3;
  int l=2;
  int n=2;

  SO3part<float> x(1,2,3,4);
  SO3part<float> y(1,2,3,4);
  SO3part<float> X(x,1);
  SO3part<float> Y(y,1);
  //print(x);
  //print(X);

  cout<<"CG product:"<<endl;
  auto z=CGproduct(x,y,1);
  auto Z=CGproduct(X,Y,1);
  print(z);
  print(Z);

  cout<<"CG product back0:"<<endl;
  auto xg=x.zeros_like();
  Gpart_add_CGproduct_back0(xg,z,y);
  auto Xg=X.zeros_like();
  Gpart_add_CGproduct_back0(Xg,Z,Y);
  print(xg);
  print(Xg);

  cout<<"CG product back1:"<<endl;
  auto yg=x.zeros_like();
  Gpart_add_CGproduct_back1(yg,z,x);
  auto Yg=Y.zeros_like();
  Gpart_add_CGproduct_back1(Yg,Z,X);
  print(yg);
  print(Yg);

  #endif

}
