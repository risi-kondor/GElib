#include "GElib_base.cpp"
#include "GElibSession.hpp"

#include "RtensorObj.hpp"
#include "CtensorObj.hpp"
#include "GroupClasses.hpp"

using namespace cnine;
using namespace GElib;

//typedef CscalarObj cscalar;
typedef RtensorObj rtensor;
typedef CtensorObj ctensor;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  CyclicGroup G(5);
  CyclicGroupIrrep rho=G.irrep(2);
  CyclicGroupElement g=G.element(2);

  cout<<rho(g)<<endl<<endl;

}

