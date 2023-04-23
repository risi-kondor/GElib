#include "GElib_base.cpp"
#include "GElibSession.hpp"
#include "SO3vecC.hpp"

using namespace cnine;
using namespace GElib;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  int b=1;
  SO3type tau({2,2,2});

  SO3vec<float> u=SO3vec<float>::gaussian(b,tau);
  cout<<u<<endl;

  SO3vec<float> v=SO3vec<float>::sequential(b,tau);
  cout<<v<<endl;

  cout<<v.part(1)<<endl;

  cout<<v+v<<endl;
  SO3vec<float> w(v);
  w.add(v);
  cout<<v<<endl;

  //Tensor<complex<float> > M=Tensor<complex<float> >::gaussian({5,5});
  //cout<<M*u<<endl;
  //cout<<u*M<<endl;

  SO3vec<float> z=CGproduct(u,v);
  cout<<z<<endl;

}
