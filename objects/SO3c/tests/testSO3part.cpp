#include "GElib_base.cpp"
#include "GElibSession.hpp"
#include "SO3partC.hpp"

using namespace cnine;
using namespace GElib;

int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  int b=3;
  int l=2;
  int n=2;

  SO3part<float> u=SO3part<float>::zero().l(l).n(n);
  SO3part<float> v=SO3part<float>::gaussian().batch(b).l(l).n(n);
  cout<<u<<endl;
  cout<<v<<endl;

  //Tensor<complex<float> > M=Tensor<complex<float> >::gaussian({5,5});
  //cout<<M*u<<endl;
  //cout<<u*M<<endl;

  //SO3part<float> w=CGproduct(u,v,2);
  //cout<<w<<endl;

  //SO3part<float> v2=SO3part<float>::gaussian(l,n);
  //cout<<CGproduct(u,v2,2)<<endl;

  //cout<<DiagCGproduct(u,v,2)<<endl;

}
