#include "GElib_base.cpp"
#include "SO3vecB.hpp"
#include "GElibSession.hpp"


using namespace cnine;
using namespace GElib;

typedef CtensorB ctensor;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  int b=1;
  SO3type tau({2,2,2});
  int maxl=2;

  SO3vecB u=SO3vecB::gaussian(1,tau);
  SO3vecB v=SO3vecB::gaussian(1,tau);
  printl("u",u)<<endl;
  printl("v",v)<<endl;

  SO3element R(fill::uniform);
  //printl("R",R);

  SO3vecB w=u.CGproduct(v,maxl);
  printl("w.rotate(R)",w.rotate(R));
  cout<<endl;

  SO3vecB uR=u.rotate(R);
  SO3vecB vR=v.rotate(R);
  SO3vecB wR=uR.CGproduct(vR,maxl);

  //printl("uR",uR);
  printl("wR",wR);
  cout<<endl;

  cout<<endl; 
}

