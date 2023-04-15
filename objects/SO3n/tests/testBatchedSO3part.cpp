#include "GElib_base.cpp"
#include "GElibSession.hpp"
#include "Tensor.hpp"
#include "BatchedSO3part.hpp"

using namespace cnine;
using namespace GElib;

int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  int b=3;
  int l=2;
  int n=2;

  BatchedSO3part<float> u=BatchedSO3part<float>::sequential(b,l,n);
  BatchedSO3part<float> v=BatchedSO3part<float>::gaussian(b,l,n);
  cout<<u.repr()<<endl;
  cout<<u<<endl;
  cout<<v<<endl;

  //Tensor<complex<float> > M=Tensor<complex<float> >::gaussian({5,5});
  //cout<<M*u<<endl;
  //cout<<u*M<<endl;

  BatchedSO3part<float> w=CGproduct(u,v,2);
  cout<<w<<endl;

  cout<<DiagCGproduct(u,v,2)<<endl;

}
