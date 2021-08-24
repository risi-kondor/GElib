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
  CyclicGroupElement g=G.element(2);
  FunctionOnGroup<CyclicGroup,rtensor> f(G,cnine::fill::gaussian);
  cout<<f<<endl;

  auto F=Fourier(f);
  cout<<F<<endl;

  auto fd=iFourier(F);
  cout<<fd<<endl;

  auto F2=F.left(g);
  auto f2d=iFourier(F2);
  cout<<f2d<<endl;

  auto F3=F.inv();
  auto f3d=iFourier(F3);
  cout<<f3d<<endl;

}

