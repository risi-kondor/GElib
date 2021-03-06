#include "GElib_base.cpp"
#include "SO3vecB_array.hpp"
#include "GElibSession.hpp"


using namespace cnine;
using namespace GElib;

typedef CtensorB ctensor;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  Gdims adims({2,1});
  SO3type tau({2,2});

  SO3vecB_array u=SO3vecB_array::gaussian(adims,tau);
  SO3vecB_array v=SO3vecB_array::gaussian(adims,tau);
  //printl("u",u)<<endl;
  //printl("v",v)<<endl;

  SO3vecB_array w=u.CGproduct(v,2);
  cout<<w<<endl;

  cout<<endl; 

#ifdef _WITH_CUDA
  SO3vecB_array uc=u.to_device(1);
  SO3vecB_array vc=v.to_device(1);

  //printl("uc",uc)<<endl;
  //printl("vc",vc)<<endl;

  SO3vecB_array wc=uc.CGproduct(vc,2);
  cout<<wc<<endl;
#endif 
  
  SO3vecB_array ug=SO3vecB_array::zeros_like(u);
  SO3vecB_array vg=SO3vecB_array::zeros_like(v);
  SO3vecB_array wg=SO3vecB_array::gaussian_like(w);

#ifdef _WITH_CUDA
  SO3vecB_array ugc=ug.to_device(1);
  SO3vecB_array vgc=vg.to_device(1);
  SO3vecB_array wgc=wg.to_device(1);
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

