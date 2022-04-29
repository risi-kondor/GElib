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

  SO3partB u=SO3partB::spharm(1,2,1,x(0),x(1),x(2));
  print("u",u);

  SO3partB ur=SO3partB::spharm(1,2,1,xr(0),xr(1),xr(2));
  print("ur",ur);

  SO3partB ur2=u.rotate(R);
  print("ur2",ur2);

  RtensorA X(Gdims({2,3}));
  X.set(0,0,x(0)); X.set(0,1,x(1)); X.set(0,2,x(2));
  X.set(1,0,x(1)); X.set(1,1,x(0)); X.set(1,2,x(2));
  SO3partB v=SO3partB::zero(2,2,1);
  v.add_spharm(X);
  print("v",v);

}
