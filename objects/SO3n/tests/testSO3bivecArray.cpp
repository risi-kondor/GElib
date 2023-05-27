#include "GElib_base.cpp"
#include "GElibSession.hpp"
#include "SO3bivecArray.hpp"

using namespace cnine;
using namespace GElib;

int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  int b=20;
  Gdims adims({10,10,10});
  SO3bitype tau({{2,2,1},{1,3,2}});

  SO3bivecArray<float> u=SO3bivecArray<float>::gaussian(b,adims,tau);
  //cout<<u<<endl;

  //SO3vecArray<float> v=SO3vecArray<float>::sequential(2,adims,tau);
  //cout<<v<<endl;

  //cout<<v.part(1)<<endl;
  //cout<<v.cell(1,1)<<endl;

  //Tensor<complex<float> > M=Tensor<complex<float> >::gaussian({5,5});
  //cout<<M*u<<endl;
  //cout<<u*M<<endl;

  SO3vecArray<float> a=CGtransform(u);
  //cout<<a<<endl;

  #ifdef _WITH_CUDA
  SO3bivecArray<float> ug(u,1);
  //cout<<ug<<endl;
  SO3vecArray<float> ag=CGtransform(ug);
  //cout<<ag<<endl;
  #endif

}
