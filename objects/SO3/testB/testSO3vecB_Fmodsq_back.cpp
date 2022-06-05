#include "GElib_base.cpp"
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
  SO3vecB z=SO3vecB::Fgaussian(b,std::min(maxl,2*L));
  SO3vecB xg=SO3vecB::Fzero(b,L);
  //printl("x",x)<<endl;

  xg.add_Fmodsq_back(z,x);
  printl("xg",xg);
  cout<<endl;

#ifdef _WITH_CUDA
  SO3vecB xc=x.to_device(1);
  SO3vecB zc=z.to_device(1);
  SO3vecB xgc=SO3vecB::Fzero(b,L,1);
  xgc.add_Fmodsq_back(zc,xc);
  printl("xgc",xgc);
  cout<<endl;
#endif

}

