#include "GElib_base.cpp"
#include "SO3Fvec.hpp"
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

  SO3Fvec x=SO3Fvec::gaussian(b,L);
  SO3Fvec z=x.Fmodsq(maxl);
  printl("z",z);
  cout<<endl;

#ifdef _WITH_CUDA
  SO3Fvec xc=x.to_device(1);
  cout<<1<<endl;
  SO3Fvec zc=xc.Fmodsq(maxl);
  cout<<2<<endl;
  printl("zc",z);
  cout<<endl;
#endif

}
