#include "GElib_base.cpp"
#include "SO3bipartArray.hpp"
#include "SO3bipart.hpp"
#include "GElibSession.hpp"


using namespace cnine;
using namespace GElib;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  int b=2;
  int l1=2;
  int l2=2;
  int nc=2;
  Gdims adims({2});

  SO3bipartArray<float> u=SO3bipartArray<float>::gaussian(b,adims,l1,l2,nc);
  SO3bipartArray<float> v=SO3bipartArray<float>::gaussian(b,adims,l1,l2,nc);
  printl("u",u)<<endl;
  printl("v",v)<<endl;

  cout<<endl; 
}
