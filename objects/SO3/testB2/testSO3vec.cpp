#include "GElib_base.cpp"
#include "SO3vec.hpp"
#include "GElibSession.hpp"


using namespace cnine;
using namespace GElib;

typedef CtensorB ctensor;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  SO3type tau({2,2});

  SO3vec u=SO3vec::gaussian(tau);
  SO3vec v=SO3vec::gaussian(tau);
  //printl("u",u)<<endl;
  //printl("v",v)<<endl;

  SO3vec w=CGproduct(u,v,2);
  cout<<w<<endl;

  cout<<endl; 

  #ifdef _WITH_CUDA
  SO3vec uc=u.to_device(1);
  SO3vec vc=v.to_device(1);

  SO3vec wc=CGproduct(uc,vc,2);
  cout<<wc<<endl;
  #endif 
  
  SO3vec ug=SO3vec::zero_like(u);
  SO3vec vg=SO3vec::zero_like(v);
  SO3vec wg=SO3vec::gaussian_like(w);

  #ifdef _WITH_CUDA
  SO3vec ugc=ug.to_device(1);
  SO3vec vgc=vg.to_device(1);
  SO3vec wgc=wg.to_device(1);
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

