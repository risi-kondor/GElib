#include "GElib_base.cpp"
#include "GElibSession.hpp"
#include "SO3vecArrayC.hpp"

using namespace cnine;
using namespace GElib;

int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  Gdims adims({2,2});
  SO3type tau({2,2,2});

  SO3vecArrayC<float> u=SO3vecArrayC<float>::zero(adims,tau);
  cout<<u<<endl;

  SO3vecArrayC<float> v=SO3vecArrayC<float>::sequential(adims,tau);
  cout<<v<<endl;

  cout<<v.part(1)<<endl;
  cout<<v.cell(1,1)<<endl;

  //Tensor<complex<float> > M=Tensor<complex<float> >::gaussian({5,5});
  //cout<<M*u<<endl;
  //cout<<u*M<<endl;

  //SO3partC<float> w=CGproduct(u,v,2);
  //cout<<w<<endl;

}
