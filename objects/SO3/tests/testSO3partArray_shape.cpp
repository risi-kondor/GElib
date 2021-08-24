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
  printl("u",u);

  printl("u.reshape(dims(4))",u.reshape(dims(4)));
  printl("u.shape(dims(4))",u.shape(dims(4)));
  printl("u.as_shape(dims(4))",u.as_shape(dims(4)));

}


