#include "GElib_base.cpp"
#include "GElibSession.hpp"
#include "GroupClasses.hpp"

using namespace cnine;
using namespace GElib;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  DihedralGroup G(5);

  for(int i=0; i<G.size(); i++) cout<<G.element(i)<<endl;
  cout<<endl;

  cout<<G.s()*G.r(2)*G.s()<<endl;

}

