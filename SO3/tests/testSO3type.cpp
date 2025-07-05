#include "GElib_base.cpp"
#include "SO3type.hpp"
#include "CGproduct.hpp"
#include "DiagCGproduct.hpp"

using namespace cnine;
using namespace GElib;

int main(int argc, char** argv){
  //GElibSession session;
  cout<<endl;

  SO3type tau({{1,3},{2,4}});
  print(tau);

  auto tau2=CGproduct(tau,tau);
  print(tau2);

}
