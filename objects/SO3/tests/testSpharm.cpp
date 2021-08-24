#include "GElib_base.cpp"
#include "SO3part.hpp"
#include "GElibSession.hpp"


using namespace cnine;
using namespace GElib;

typedef CscalarObj Cscalar;
typedef CtensorObj Ctensor;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  Gtensor<float> x({3},fill::gaussian);
  SO3element R(fill::uniform);
  Gtensor<float> xr=R(x);

  SO3part u=SO3part::spharm(2,x(0),x(1),x(2));
  print("u",u);

  SO3part ur=SO3part::spharm(2,xr(0),xr(1),xr(2));
  print("ur",ur);

  SO3part ur2=u.rotate(R);
  print("ur2",ur2);


}
