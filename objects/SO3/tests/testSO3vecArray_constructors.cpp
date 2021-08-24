#include "GElib_base.cpp"
#include "SO3vecArray.hpp"
#include "GElibSession.hpp"


using namespace cnine;
using namespace GElib;

typedef CscalarObj cscalar;
typedef CtensorObj ctensor;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  Gdims adims({2,2});
  SO3type tau({1,1,1});

  SO3vecArray u0(adims,tau,fill::gaussian);
  printl("u0",u0);

  SO3vec p(tau,fill::gaussian);
  SO3vecArray u1(adims,p);
  printl("u1",u1); 

  SO3vecArray u2(adims,tau,[&adims](const Gindex& ix, const int l, const int i, const int m){
      return complex<float>(ix(adims),2);}); // check this 
  printl("u2",u2);

}


