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
  SO3type tau({2,2,2});

  SO3vecB u=SO3vecB::gaussian(1,tau);
  SO3vecB v=SO3vecB::gaussian(1,tau);
  printl("u",u)<<endl;
  printl("v",v)<<endl;

  SO3vecB w=u.CGproduct(v,2);
  cout<<w<<endl;

  cout<<endl; 

#ifdef _WITH_CUDA
  SO3vecB uc=u.to_device(1);
  SO3vecB vc=v.to_device(1);

  printl("uc",uc)<<endl;
  printl("vc",vc)<<endl;

  SO3vecB wc=uc.CGproduct(vc,2);
  cout<<wc<<endl;
#endif 
  
  SO3vecB ug=SO3vecB::zeros_like(u);
  SO3vecB vg=SO3vecB::zeros_like(v);
  SO3vecB wg=SO3vecB::gaussian_like(w);

#ifdef _WITH_CUDA
  SO3vecB ugc=ug.to_device(1);
  SO3vecB vgc=vg.to_device(1);
  SO3vecB wgc=wg.to_device(1);
#endif

  cout<<"----------- back0 -----------------------"<<endl;

  ug.add_CGproduct_back0(wg,v);
  printl("ug",ug);

#ifdef _WITH_CUDA
  ugc.add_CGproduct_back0(wgc,vc);
  printl("ugc",ugc);
#endif

  cout<<"----------- back1 -----------------------"<<endl;

  vg.add_CGproduct_back0(wg,u);
  printl("vg",vg);

#ifdef _WITH_CUDA
  vgc.add_CGproduct_back0(wgc,uc);
  printl("vgc",vgc);
#endif 

}

