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

  SO3vecB x=SO3vecB::Fzero(b,L);
  SO3vecB y=SO3vecB::Fgaussian(b,L);
  SO3vecB z=SO3vecB::Fgaussian(b,std::min(maxl,2*L));
  //printl("x",x)<<endl;
  //printl("y",y)<<endl;

  x.add_Fproduct_back0(z,y);
  printl("x",x);
  cout<<endl;

#ifdef _WITH_CUDA
  SO3vecB xc=SO3vecB::Fzero(b,L,1);
  SO3vecB yc=y.to_device(1);
  SO3vecB zc=z.to_device(1);

  xc.add_Fproduct_back0(zc,yc);
  printl("xc",xc);
  cout<<endl;
#endif

}

