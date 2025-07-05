#include "GElib_base.cpp"
#include "O3type.hpp"
#include "CGproduct.hpp"
#include "DiagCGproduct.hpp"

using namespace cnine;
using namespace GElib;

int main(int argc, char** argv){
  cout<<endl;

  O3type tau({{{1,1},3},{{2,-1},4}});
  print(tau);

  auto tau2=CGproduct(tau,tau);
  print(tau2);

  auto tau3=DiagCGproduct(tau,tau);
  print(tau3);

}
