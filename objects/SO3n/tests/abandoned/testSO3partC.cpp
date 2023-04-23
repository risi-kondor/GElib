#include "GElib_base.cpp"
#include "SO3partB_array.hpp"
#include "GElibSession.hpp"
#include "Tensor.hpp"
#include "SO3partC.hpp"

using namespace cnine;
using namespace GElib;

typedef RtensorA rtensor;
typedef CtensorB ctensor;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  int n=4;
  int l=2;
  int nc=2;
  Gdims adims({n});

  SO3partC<float> u=SO3partC<float>::sequential(l,nc);
  SO3partC<float> v=SO3partC<float>::gaussian(l,nc);
  cout<<u.repr()<<endl;
  cout<<u<<endl;
  cout<<v<<endl;

  //Tensor<complex<float> > M=Tensor<complex<float> >::gaussian({5,5});
  //cout<<M*u<<endl;
  //cout<<u*M<<endl;

  SO3partC<float> w=CGproduct(u,v,2);
  cout<<w<<endl;

  cout<<DiagCGproduct(u,v,2)<<endl;

}
