#include "GElib_base.cpp"
#include "GElibSession.hpp"
#include "SO3vecC.hpp"

using namespace cnine;
using namespace GElib;

int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  SO3type tau({2,2,2});

  SO3vecC<float> u=SO3vecC<float>::zero(tau);
  cout<<u<<endl;

  SO3vecC<float> v=SO3vecC<float>::sequential(tau);
  cout<<v<<endl;

  cout<<v.part(1)<<endl;

  cout<<v+v<<endl;
  SO3vecC<float> w(v);
  w.add(v);
  cout<<v<<endl;

  //Tensor<complex<float> > M=Tensor<complex<float> >::gaussian({5,5});
  //cout<<M*u<<endl;
  //cout<<u*M<<endl;

  SO3vecC<float> w=CGproduct(u,v);
  cout<<w<<endl;

}
