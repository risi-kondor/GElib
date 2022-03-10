#include "GElib_base.cpp"
#include "SO3partB.hpp"
#include "GElibSession.hpp"


using namespace cnine;
using namespace GElib;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  const int b=2;
  const int n=2;
  const int l0=2;
  const int l1=3;
  const int l=3;

  SO3partB u=SO3partB::gaussian(b,l0,n);
  SO3partB v=SO3partB::gaussian(b,l1,n);
  printl("u",u)<<endl;
  printl("v",v)<<endl;

  SO3element R(fill::uniform);
  //printl("R",R);

  SO3partB w=u.CGproduct(v,l);
  printl("w.rotate(R)",w.rotate(R));
  cout<<endl;

  SO3partB uR=u.rotate(R);
  SO3partB vR=v.rotate(R);
  SO3partB wR=uR.CGproduct(vR,l);

  //printl("uR",uR);
  printl("wR",wR);
  cout<<endl;

  cout<<endl; 
}
