#include "GElib_base.cpp"
#include "SO3partArray.hpp"
#include "GElibSession.hpp"


using namespace cnine;
using namespace GElib;

typedef CscalarObj cscalar;
typedef CtensorObj ctensor;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  Gdims adims({2,2});

  SO3partArray u0(adims,2,2,fill::gaussian);
  printl("u0",u0);

  SO3part p(2,2,fill::gaussian);
  SO3partArray u1(adims,p);
  printl("u1",u1); 

  SO3partArray u2(adims,2,2,[&adims](const Gindex& ix, const int i, const int m){
      return complex<float>(ix(adims),m);});
  printl("u2",u2);

  SO3partArray uv(adims,2,2,fill::view,u2.arr,u2.arrc);
  printl("uv",uv);

}


