#include "GElib_base.cpp"
#include "SO3partB.hpp"
#include "GElibSession.hpp"


using namespace cnine;
using namespace GElib;

extern GElib::SO3_CGbank SO3_cgbank;

int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  const int b=1;
  const int n=1;
  const int l0=3;
  const int l1=3;
  const int l=6;

  auto& C=SO3_cgbank.getf(CGindex(l0,l1,l));
  cout<<C<<endl;

  SO3element R(fill::uniform);
  //SO3element R(0,0,0);
  //printl("R",R);
  CtensorB D(WignerMatrix<float>(l,R));
  cout<<D<<endl;

  SO3partB u=SO3partB::gaussian(b,l0,n);
  SO3partB v=SO3partB::gaussian(b,l1,n);
  printl("u",u)<<endl;
  printl("v",v)<<endl;


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
