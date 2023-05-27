#include "GElib_base.cpp"
#include "GElibSession.hpp"
#include "SO3bivec.hpp"

using namespace cnine;
using namespace GElib;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  int b=2;
  //SO3bitype tau({{1,1,1}});
  SO3bitype tau({{2,2,1},{1,3,2}});
  //SO3bitype tau({{2,2},1},{{1,3},2});

  SO3bivec<float> u=SO3bivec<float>::gaussian(b,tau);
  //cout<<u<<endl;

  SO3bivec<float> v=SO3bivec<float>::sequential(b,tau);
  //cout<<v<<endl;

  //cout<<v.part(1)<<endl;

  //cout<<v+v<<endl;
  SO3bivec<float> w(v);
  w.add(v);
  //cout<<v<<endl;

  //Tensor<complex<float> > M=Tensor<complex<float> >::gaussian({5,5});
  //cout<<M*u<<endl;
  //cout<<u*M<<endl;

  SO3vec<float> a=CGtransform(u);
  cout<<a<<endl;

  #ifdef _WITH_CUDA
  SO3bivec<float> ug(u,1);
  //cout<<ug<<endl;
  SO3vec<float> ag=CGtransform(ug);
  cout<<ag<<endl;
  #endif

}
