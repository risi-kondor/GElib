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
  SO3vecB y=SO3vecB::Fzero(b,L);
  SO3vecB z=SO3vecB::Fgaussian(b,std::min(maxl,2*L));
  //printl("x",x)<<endl;
  //printl("y",y)<<endl;

  y.add_Fproduct_back1(z,x);
  printl("y",y);
  cout<<endl;

#ifdef _WITH_CUDA
  SO3vecB xc=x.to_device(1);
  SO3vecB yc=SO3vecB::Fzero(b,L,1);
  SO3vecB zc=z.to_device(1);

  yc.add_Fproduct_back0(zc,xc);
  printl("yc",yc);
  cout<<endl;
#endif

}

