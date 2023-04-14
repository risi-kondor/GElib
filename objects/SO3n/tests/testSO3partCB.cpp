#include "GElib_base.cpp"
#include "GElibSession.hpp"
#include "Tensor.hpp"
#include "SO3partCB.hpp"

using namespace cnine;
using namespace GElib;

int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  int b=3;
  int n=4;

  SO3partCB<float> u=SO3partCB<float>::sequential(b,2,n);
  SO3partCB<float> v=SO3partCB<float>::sequential(b,2,n);
  cout<<u.repr()<<endl;
  cout<<u<<endl;

  //Tensor<complex<float> > M=Tensor<complex<float> >::gaussian({5,5});
  //cout<<M*u<<endl;
  //cout<<u*M<<endl;

  SO3partCB<float> w=CGproduct(u,v,2);
  cout<<w<<endl;

}
