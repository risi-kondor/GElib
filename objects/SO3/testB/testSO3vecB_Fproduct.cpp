#include "GElib_base.cpp"
//#include "SO3Fvec.hpp"
#include "SO3vecB.hpp"
#include "GElibSession.hpp"


using namespace cnine;
using namespace GElib;

typedef CtensorB ctensor;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  int b=2;
  int L1=2;
  int L2=3;
  int maxl=3;

  SO3vecB x=SO3vecB::Fgaussian(b,L1);
  SO3vecB y=SO3vecB::Fgaussian(b,L2);
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
  SO3vecB zcb=xc.FproductB(yc,maxl);
  printl("zcb",zcb);
  cout<<endl;
#endif

}

