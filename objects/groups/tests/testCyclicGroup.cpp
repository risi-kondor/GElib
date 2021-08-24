#include "GElib_base.cpp"
#include "GElibSession.hpp"
#include "GroupClasses.hpp"

using namespace cnine;
using namespace GElib;


int main(int argc, char** argv){
  GElibSession session;
  //CombinatorialClasses combi_classes;
  cout<<endl;

  CyclicGroup G(5);

  for(int i=0; i<G.size(); i++) cout<<G.element(i)<<endl;

}

