#include "GElib_base.cpp"
#include "SO3part.hpp"
#include "GElibSession.hpp"
#include "CtensorObj_funs.hpp"

using namespace cnine;
using namespace GElib;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  const int lx=2;
  const int ly=3;
  const int n=2;
  const int l=3;

  SO3part u(lx,n,fill::gaussian);
  SO3part v(ly,n,fill::gaussian);
  printl("u",u)<<endl;
  printl("v",v)<<endl;
  SO3element R(fill::uniform);
  //printl("R",R);

  SO3part w=CGproduct(u,v,l);
  printl("w.rotate(R)",w.rotate(R));
  cout<<endl;

  SO3part uR=u.rotate(R);
  SO3part vR=v.rotate(R);
  SO3part wR=CGproduct(uR,vR,l);

  //printl("uR",uR);
  printl("wR",wR);
  cout<<endl;

}
