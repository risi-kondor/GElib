#include "GElib_base.cpp"
#include "GElibSession.hpp"
#include "GroupClasses.hpp"

using namespace cnine;
using namespace GElib;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  CyclicGroup G1(3);
  CyclicGroup G2(4);

  auto G=G1*G2;
  auto rho=G.irrep(3);

  for(int i=0; i<G.size(); i++) 
    cout<<rho(i)<<endl;

}

