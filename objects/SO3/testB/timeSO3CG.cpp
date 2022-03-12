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
  int l1=3;
  int l2=3;
  int l=4;
  int n1=2;
  int n2=2;

  SO3partB u=SO3partB::gaussian(b,l1,n1);
  SO3partB v=SO3partB::gaussian(b,l2,n2);
  //printl("u",u)<<endl;
  //printl("v",v)<<endl;

  SO3partB w=u.CGproduct(v,l);
  cout<<w<<endl;

#ifdef _WITH_CUDA
  auto ug=u.to_device(1);
  auto vg=v.to_device(1);
  auto wg=ug.CGproduct(vg,l);

  cout<<wg<<endl;
#endif 

  cout<<endl; 

}

