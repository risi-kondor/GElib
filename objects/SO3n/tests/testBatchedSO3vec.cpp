#include "GElib_base.cpp"
#include "GElibSession.hpp"
#include "BatchedSO3vec.hpp"

using namespace cnine;
using namespace GElib;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  int b=2;
  SO3type tau({2,2,2});

  BatchedSO3vec<float> u=BatchedSO3vec<float>::zero(b,tau);
  cout<<u<<endl;

  BatchedSO3vec<float> v=BatchedSO3vec<float>::sequential(b,tau);
  cout<<v<<endl;

  cout<<v.part(1)<<endl;

  cout<<v+v<<endl;
  BatchedSO3vec<float> w(v);
  w.add(v);
  cout<<v<<endl;

  //Tensor<complex<float> > M=Tensor<complex<float> >::gaussian({5,5});
  //cout<<M*u<<endl;
  //cout<<u*M<<endl;

  //BatchedSO3vec<float> w=CGproduct(u,v);
  //cout<<w<<endl;

}
