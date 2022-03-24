#include "GElib_base.cpp"
#include "SO3vec.hpp"
#include "GElibSession.hpp"


using namespace cnine;
using namespace GElib;

typedef CtensorB ctensor;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  int b=1;
  SO3type tau({2,2});
  int maxl=2;

  SO3vec x=SO3vecB::gaussian(b,tau);
  cout<<CGproduct(x,x,maxl)<<endl;

  cout<<CGsquare(x,maxl)<<endl;

#ifdef _WITH_CUDA
  SO3vec xc=x.to_device(1);
  cout<<endl;
#endif

}

