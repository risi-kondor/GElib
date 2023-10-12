#include "GElib_base.cpp"
#include "GElibSession.hpp"
#include "SO3partC.hpp"
#include "Ltensor.hpp"

using namespace cnine;
using namespace GElib;

int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  int b=3;
  int l=2;
  int n=2;

  SO3part<float> u=SO3part<float>::gaussian().l(l).n(n);
  SO3part<float> v=SO3part<float>::gaussian().batch(b).l(l).n(n);
  cout<<u<<endl;
  cout<<v<<endl;

  Ltensor<complex<float> > M=Ltensor<complex<float> >::gaussian().dims({2,3});
  cout<<M<<endl;
  //cout<<M*u<<endl;
  cout<<u*M<<endl;

  SO3part<float> w=CGproduct(u,u,2);
  cout<<w<<endl;


}
