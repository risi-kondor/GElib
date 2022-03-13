#include "GElib_base.cpp"
#include "SO3vecB.hpp"
#include "GElibSession.hpp"


using namespace cnine;
using namespace GElib;

typedef CtensorB ctensor;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  int b=10;
  int l1=2;
  int l2=2;
  int l=3;
  int n1=20;
  int n2=20;

#ifdef _WITH_CUDA
  SO3partB u0=SO3partB::gaussian(1,l1,1,1);
  SO3partB v0=SO3partB::gaussian(1,l2,1,1);
  SO3partB w0=u0.CGproduct(v0,l);
#endif

  SO3partB u=SO3partB::gaussian(b,l1,n1);
  SO3partB v=SO3partB::gaussian(b,l2,n2);
  //cout<<u.dims<<endl;
  //cout<<u.strides(0)<<u.strides(1)<<u.strides(2)<<endl;
  printl("u",u)<<endl;
  printl("v",v)<<endl;

  cout<<"Starting CPU"<<endl;
  SO3partB w=u.CGproduct(v,l);
  //cout<<w<<endl;
  cout<<"."<<endl;

#ifdef _WITH_CUDA
  SO3partB ug=u.to_device(1);
  SO3partB vg=v.to_device(1);
  cout<<"Starting CPU"<<endl;
  SO3partB wg=ug.CGproduct(vg,l);

  //cout<<wg<<endl;
  cout<<"."<<endl;
#endif 

  cout<<endl; 

}

