#include "GElib_base.cpp"
#include "SO3Fvec.hpp"
#include "SO3vecB.hpp"
#include "GElibSession.hpp"


using namespace cnine;
using namespace GElib;

typedef CtensorB ctensor;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  int b=1;
  int L=2;
  int maxl=2;

  SO3vecB x=SO3vecB::Fgaussian(b,L);
  SO3vecB y=SO3vecB::Fgaussian(b,L);
  //printl("x",x)<<endl;
  //printl("y",y)<<endl;

  SO3vecB z=x.Fproduct(y,maxl);
  printl("z",z);
  cout<<endl;

#ifdef _WITH_CUDA
  SO3vecB xc=x.to_device(1);
  SO3vecB yc=y.to_device(1);
  SO3vecB zc=xc.Fproduct(yc,maxl);
  printl("zc",zc);
  cout<<endl;
#endif

}
