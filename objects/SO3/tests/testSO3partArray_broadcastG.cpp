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

  SO3partArray u(adims,2,2,fill::gaussian,dev);


  // Broadcasting a single SO3part to an SO3partArray
  SO3part p(2,2,fill::gaussian,dev);
  SO3partArray pp(adims,p);
  printl("pp",pp); 

  // Duplicating an SO3partArray twice along different directions 
  printl("u.broaden(0,2)",u.broaden(0,2));
  printl("u.broaden(1,2)",u.broaden(1,2));
  printl("u.broaden(2,2)",u.broaden(2,2));

  // Reducing an SO3partArray along different dimensions 
  printl("u.reduce(0)",u.reduce(0));
  printl("u.reduce(1)",u.reduce(1));

  printl("u.reduce()",u.reduce());

}


