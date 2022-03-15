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
  int l2=2;
  int l=3;
  int n1=2;
  int n2=2;
  int niter=1;

  SO3partB u=SO3partB::gaussian(b,l1,n1);
  SO3partB v=SO3partB::zero(b,l2,n2);
  SO3partB w=SO3partB::gaussian(b,l,n1*n2);
  //printl("u",u)<<endl;
  //printl("w",w)<<endl;

  //cout<<"Starting CPU"<<endl;
  v.add_CGproduct_back1(w,u);
  cout<<v<<endl;

#ifdef _WITH_CUDA
  SO3partB ug=u.to_device(1);
  SO3partB vg=SO3partB::zero(b,l2,n2,1);
  SO3partB wg=w.to_device(1);
  //cout<<"Starting GPU"<<endl;
  vg.add_CGproduct_back1(wg,ug);
  cout<<vg<<endl;
#endif 

  cout<<endl; 

}

