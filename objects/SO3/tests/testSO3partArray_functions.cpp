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

  SO3partArray u(adims,2,2,fill::gaussian);
  SO3partArray v(adims,2,2,fill::gaussian);

  SO3part p(2,2,fill::gaussian);


  //printl("inp(u,v)",inp(u,v));
  //printl("inp(u,p)",inp(u,p));
  //printl("inp(p,v)",inp(p,v));

  cout<<endl; 
}

