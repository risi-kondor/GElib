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
  //printl("z",z);
  cout<<endl;

#ifdef _WITH_CUDA
  SO3Fvec xc=x.to_device(1);
  SO3Fvec zc=xc.Fmodsq(maxl);
  //printl("zc",zc);
  cout<<endl;
#endif

  SO3Fvec xg=SO3Fvec::zeros_like(x);
//  SO3Fvec xgd=SO3Fvec::zeros_like(x);
  SO3Fvec zg=SO3Fvec::gaussian_like(z);

#ifdef _WITH_CUDA
  SO3Fvec xgc=xg.to_device(1);
 // SO3Fvec xgdc=xgd.to_device(1);
  SO3Fvec zgc=zg.to_device(1);
#endif

  xg.add_Fmodsq_back(zg,x);
//  xgd.add_Fproduct_back1(zg,x);
  printl("xg",xg);
//  printl("xgd",xgd);

#ifdef _WITH_CUDA
  xgc.add_Fmodsq_back(zgc,xc);
//  xgdc.add_Fproduct_back1(zgc,xc);
  printl("xgc",xgc);
//  printl("xgdc",xgdc);
#endif


}
