#include "GElib_base.cpp"
#include "SO3partB_array.hpp"
#include "GElibSession.hpp"
#include "RtensorA.hpp"

using namespace cnine;
using namespace GElib;

typedef RtensorA rtensor;
typedef CtensorB ctensor;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  int n=4;
  int B=2;
  Gdims adims({n});

  SO3partB_array u=SO3partB_array::gaussian(adims,2,2);

  rtensor M=rtensor::zero({n,n});
  M.set(0,1,1.0);
  M.set(0,3,1.0);
  M.set(2,2,1.0);
  cout<<M<<endl;

  Rmask1 mask=Rmask1::matrix(M.view2());
  cout<<mask<<endl;
  cout<<mask.inv()<<endl;
  
  SO3partB_array w=u.gather(mask);
}
