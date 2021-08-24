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
  SO3type tau({1,1});

  SO3vecArray u(adims,tau,fill::gaussian);


  // Broadcasting a single SO3vec to an SO3vecArray
  SO3vec p(tau,fill::gaussian);
  SO3vecArray pp(adims,p);
  printl("pp",pp); 

  // Duplicating an SO3vecArray along different directions 
  printl("u.repeat(0,2)",u.repeat(0,2));
  printl("u.repeat(1,2)",u.repeat(1,2));
  printl("u.repeat(2,2)",u.repeat(2,2));

  // Reducing an SO3vecArray along different dimensions 
  printl("u.reduce(0)",u.reduce(0));
  printl("u.reduce(1)",u.reduce(1));

}


