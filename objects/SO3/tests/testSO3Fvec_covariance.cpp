#include "GElib_base.cpp"
#include "SO3Fvec.hpp"
#include "GElibSession.hpp"


using namespace cnine;
using namespace GElib;

typedef CtensorB ctensor;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  int b=1;
  int L=2;
  int maxl=2;

  SO3Fvec u=SO3Fvec::gaussian(b,L);
  SO3Fvec v=SO3Fvec::gaussian(b,L);
  printl("u",u)<<endl;
  printl("v",v)<<endl;

  SO3element R(fill::uniform);
  //printl("R",R);

  SO3Fvec w=u.Fproduct(v,maxl);
  //printl("w",w);
  printl("w.rotate(R)",w.rotate(R));
  cout<<endl;

  SO3Fvec uR=u.rotate(R);
  SO3Fvec vR=v.rotate(R);
  SO3Fvec wR=uR.Fproduct(vR,maxl);

  //printl("uR",uR);
  printl("wR",wR);
  cout<<endl;

  cout<<endl; 
}
