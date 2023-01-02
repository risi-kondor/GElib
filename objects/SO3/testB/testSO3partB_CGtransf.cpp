#include "GElib_base.cpp"
#include "SO3vecB.hpp"
#include "SO3part_addCGtransfFn.hpp"
//#include "SO3Fvec.hpp"
#include "GElibSession.hpp"


using namespace cnine;
using namespace GElib;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  int b=1;
  int l1=1;
  int l2=1;
  int l=1;
  int c=1;

  CtensorB x=CtensorB::gaussian({b,2*l1+1,2*l2+1,c});
  SO3partB r=SO3partB::zero(b,l,c);

  SO3part_addCGtransfFn()(r,x.view4());
  cout<<r<<endl;

#ifdef _WITH_CUDA
  SO3partB xg=x.to_device(1);
  SO3partB rg=SO3partB::zero(b,l,c,1);

  SO3part_addCGtransfFn()(r,x.view4());
  cout<<rg<<endl;
#endif 

}

