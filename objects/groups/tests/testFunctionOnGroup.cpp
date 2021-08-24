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

  FunctionOnGroup<CyclicGroup,rtensor> f(G,cnine::fill::gaussian);
  cout<<f<<endl;

  CyclicGroupElement g=G.element(2);
  cout<<f(g)<<endl<<endl;

  cout<<f.left(g)<<endl;

  cout<<f.right(g)<<endl;

  cout<<f.inv()<<endl;

}

