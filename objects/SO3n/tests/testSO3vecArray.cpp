#include "GElib_base.cpp"
#include "GElibSession.hpp"
#include "SO3vecArrayC.hpp"

using namespace cnine;
using namespace GElib;

int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  int b=2;
  Gdims adims({2});
  SO3type tau({2,2});

  SO3vecArray<float> u=SO3vecArray<float>::gaussian(2,adims,tau);
  cout<<u<<endl;

  SO3vecArray<float> v=SO3vecArray<float>::sequential(2,adims,tau);
  cout<<v<<endl;

  //cout<<v.part(1)<<endl;
  //cout<<v.cell(1,1)<<endl;

  //Tensor<complex<float> > M=Tensor<complex<float> >::gaussian({5,5});
  //cout<<M*u<<endl;
  //cout<<u*M<<endl;

  SO3vecArray<float> w=CGproduct(u,v,2);
  cout<<w<<endl;

}
