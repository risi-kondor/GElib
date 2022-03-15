#include "GElib_base.cpp"
#include "SO3vecB.hpp"
#include "GElibSession.hpp"


using namespace cnine;
using namespace GElib;

typedef CtensorB ctensor;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  int b=2;
  int l1=1;
  int l2=1;
  int l=1;
  int n1=2;
  int n2=2;
  int niter=1;

#ifdef _WITH_CUDA
//  SO3partB u0=SO3partB::gaussian(1,l1,1,1);
//  SO3partB v0=SO3partB::gaussian(1,l2,1,1);
//  SO3partB w0=u0.CGproduct(v0,l);
#endif

  SO3partB u=SO3partB::zero(b,l1,n1);
  SO3partB v=SO3partB::gaussian(b,l2,n2);
  SO3partB w=SO3partB::gaussian(b,l,n1*n2);
  //printl("u",u)<<endl;
  //printl("w",w)<<endl;

  //cout<<"Starting CPU"<<endl;
  for(int i=0; i<niter; i++){
    u.add_CGproduct_back0(w,v);
    cout<<u<<endl;
  }

#ifdef _WITH_CUDA
  SO3partB ug=SO3partB::zero(b,l1,n1,1);
  SO3partB vg=v.to_device(1);
  SO3partB wg=w.to_device(1);
  cout<<"Starting GPU"<<endl;
  for(int i=0; i<niter; i++){
    ug.add_CGproduct_back0(wg,vg);
    cout<<ug<<endl;
  }
#endif 

  cout<<endl; 

}

