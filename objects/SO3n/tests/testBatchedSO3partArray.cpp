#include "GElib_base.cpp"
#include "BatchedSO3part.hpp"
#include "BatchedSO3partArray.hpp"
#include "SO3partC.hpp"
#include "GElibSession.hpp"


using namespace cnine;
using namespace GElib;

typedef CtensorB ctensor;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  int b=2;
  int l=2;
  int nc=2;
  Gdims adims({2});

  BatchedSO3partArray<float> u=BatchedSO3partArray<float>::gaussian(b,adims,l,nc);
  BatchedSO3partArray<float> v=BatchedSO3partArray<float>::gaussian(b,adims,l,nc);
  printl("u",u)<<endl;
  printl("v",v)<<endl;

  BatchedSO3partArray<float> w=CGproduct(u,v,2);
  cout<<w<<endl;

  cout<<DiagCGproduct(u,v,2)<<endl;

  cout<<u(0)<<endl;

  BatchedSO3part<float> a=BatchedSO3part<float>::sequential(b,2,3);
  cout<<BatchedSO3partArray<float>({2,2},a)<<endl;

  cout<<endl; 
}
