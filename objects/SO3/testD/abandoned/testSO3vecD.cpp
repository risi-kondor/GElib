#include "GElib_base.cpp"
#include "SO3vecD.hpp"
#include "GElibSession.hpp"


using namespace cnine;
using namespace GElib;

typedef CtensorB ctensor;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  int N=2;
  int B=1;
  SO3type tau({2,2});

  SO3vecD u=SO3vecD::gaussian(N,B,tau);
  SO3vecD v=SO3vecD::gaussian(N,B,tau);
  //printl("u",u)<<endl;
  //printl("v",v)<<endl;

  SO3vecD w=u.CGproduct(v,2);
  cout<<w<<endl;

  cout<<endl; 

#ifdef _WITH_CUDA
  SO3vecD uc=u.to_device(1);
  SO3vecD vc=v.to_device(1);

  //printl("uc",uc)<<endl;
  //printl("vc",vc)<<endl;

  SO3vecD wc=uc.CGproduct(vc,2);
  cout<<wc<<endl;
#endif 
  
  SO3vecD ug=SO3vecD::zeros_like(u);
  SO3vecD vg=SO3vecD::zeros_like(v);
  SO3vecD wg=SO3vecD::gaussian_like(w);

#ifdef _WITH_CUDA
  SO3vecD ugc=ug.to_device(1);
  SO3vecD vgc=vg.to_device(1);
  SO3vecD wgc=wg.to_device(1);
#endif

  cout<<"----------- back0 -----------------------"<<endl;

  ug.add_CGproduct_back0(wg,v);
  printl("ug",ug);

#ifdef _WITH_CUDA
  ugc.add_CGproduct_back0(wgc,vc);
  printl("ugc",ugc);
#endif

  cout<<"----------- back1 -----------------------"<<endl;

  vg.add_CGproduct_back1(wg,u);
  printl("vg",vg);

#ifdef _WITH_CUDA
  vgc.add_CGproduct_back1(wgc,uc);
  printl("vgc",vgc);
#endif 

}

