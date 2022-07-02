#include "GElib_base.cpp"
#include "SO3mvec.hpp"
#include "SO3mweights.hpp"
#include "GElibSession.hpp"


using namespace cnine;
using namespace GElib;

typedef CtensorB ctensor;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  int b=1;
  int k=2;
  SO3type tau({2,2});
  SO3type tau2({1,3});

  SO3mvec u=SO3mvec::gaussian(b,k,tau);
  SO3mvec v=SO3mvec::gaussian(b,k,tau);
  //printl("u",u)<<endl;
  //printl("v",v)<<endl;

  SO3mvec w=CGproduct(u,v,2);
  cout<<w<<endl;

  cout<<endl; 

  #ifdef _WITH_CUDA
  SO3mvec uc=u.to_device(1);
  SO3mvec vc=v.to_device(1);

  SO3mvec wc=CGproduct(uc,vc,2);
  cout<<wc<<endl;
  #endif 
  
  SO3mvec ug=SO3mvec::zeros_like(u);
  SO3mvec vg=SO3mvec::zeros_like(v);
  SO3mvec wg=SO3mvec::gaussian_like(w);

  #ifdef _WITH_CUDA
  SO3mvec ugc=ug.to_device(1);
  SO3mvec vgc=vg.to_device(1);
  SO3mvec wgc=wg.to_device(1);
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

  {
    cout<<"----------- SO3mweights -----------------------"<<endl;
  
    SO3mweights W=SO3mweights::gaussian(k,tau,tau2);
    cout<<W<<endl;

    SO3mvec u2=u*W;
    cout<<u2<<endl;

    cout<<1<<endl;
    auto ug=SO3mvec::zeros_like(u);
    ug.add_mprod_back0(u2,W);

    cout<<2<<endl;
    auto Wg=SO3mweights::zeros_like(W);
    u2.add_mprod_back1_into(Wg,u);

  }
}

