#include "GElib_base.cpp"
#include "GElibSession.hpp"
#include "Tensor.hpp"
#include "SO3bipart.hpp"

using namespace cnine;
using namespace GElib;

int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  int b=3;
  int l1=2;
  int l2=3;
  int n=2;

  SO3bipart<float> u=SO3bipart<float>::sequential(b,l1,l2,n);
  SO3bipart<float> v=SO3bipart<float>::gaussian(b,l1,l2,n);
  cout<<u.repr()<<endl;
  cout<<u<<endl;
  cout<<v<<endl;

  SO3part<float> w=CGtransform(u,2);
  printl("w=",w);

}
