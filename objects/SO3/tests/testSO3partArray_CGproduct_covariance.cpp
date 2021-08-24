#include "GElib_base.cpp"
#include "SO3partArray.hpp"
#include "GElibSession.hpp"
#include "CtensorObj_funs.hpp"

using namespace cnine;
using namespace GElib;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  const Gdims adims({2,2});
  const int lx=2;
  const int ly=3;
  const int n=2;
  const int l=3;


  SO3partArray u(adims,lx,n,fill::gaussian);
  SO3partArray v(adims,ly,n,fill::gaussian);
  printl("u",u)<<endl;
  printl("v",v)<<endl;
  SO3element R(fill::uniform);
  //printl("R",R);

  SO3partArray w=CGproduct(u,v,l);
  printl("w.rotate(R)",w.rotate(R));
  cout<<endl;

  SO3partArray uR=u.rotate(R);
  SO3partArray vR=v.rotate(R);
  SO3partArray wR=CGproduct(uR,vR,l);

  //printl("uR",uR);
  printl("wR",wR);
  cout<<endl;

}
