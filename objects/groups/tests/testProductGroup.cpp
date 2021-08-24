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

  //ProductGroup<CyclicGroup,CyclicGroup> G(G1,G2);
  auto G=G1*G2;

  cout<<G.identity()<<endl<<endl;

  for(int i=0; i<G.size(); i++) 
    cout<<G.element(i)<<endl;

}

