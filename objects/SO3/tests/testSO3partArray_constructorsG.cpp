#include "GElib_base.cpp"
#include "SO3partArray.hpp"
#include "GElibSession.hpp"


using namespace cnine;
using namespace GElib;

typedef CscalarObj cscalar;
typedef CtensorObj ctensor;


int main(int argc, char** argv){
  GElibSession session;
  device dev=deviceid::GPU0;
  cout<<endl;

  Gdims adims({2,2});

  SO3partArray u0(adims,2,2,fill::gaussian,dev);
  printl("u0",u0);

  SO3part p(2,2,fill::gaussian,dev);
  SO3partArray u1(adims,p,dev);
  printl("u1",u1); 

  SO3partArray u2(adims,2,2,[&adims](const Gindex& ix, const int i, const int m){
      return complex<float>(ix(adims),m);},dev);
  printl("u2",u2);


}


