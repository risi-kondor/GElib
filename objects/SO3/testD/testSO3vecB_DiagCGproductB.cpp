#include "GElib_base.cpp"
#include "SO3vecB_array.hpp"
#include "GElibSession.hpp"


using namespace cnine;
using namespace GElib;

typedef CtensorB ctensor;

extern GElib::GElibConfig* gelib_config;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  int b=1;
  Gdims adims({1});
  SO3type tau({4,4,4});
  //SO3type tau({32,32,32});
  //SO3type tau({128,128,128});
  //SO3type tau({256,256,256});

  SO3vecB_array u=SO3vecB_array::gaussian(b,adims,tau);
  SO3vecB_array v=SO3vecB_array::gaussian(b,adims,tau);
  //printl("u",u)<<endl;
  //printl("v",v)<<endl;

  SO3vecB_array w=u.DiagCGproduct(v,2);
  cout<<w<<endl;

  SO3vecB_array wB=u.DiagCGproductB(v,2);
  cout<<wB<<endl;
    
  cout<<endl; 
    
#ifdef _WITH_CUDA
  SO3vecB_array uc=u.to_device(1);
  SO3vecB_array vc=v.to_device(1);
  //printl("uc",uc)<<endl;
  //printl("vc",vc)<<endl;
  
  SO3vecB_array wc=uc.DiagCGproduct(vc,2);
  //cout<<wc<<endl;

  SO3vecB_array wcB=uc.DiagCGproductB(vc,2);
  //cout<<wcB<<endl;
#endif 
  

  SO3vecB_array ug=SO3vecB_array::zeros_like(u);
  SO3vecB_array vg=SO3vecB_array::zeros_like(v);
  SO3vecB_array wg=SO3vecB_array::gaussian_like(w);
  SO3vecB_array ugB=SO3vecB_array::zeros_like(u);
  SO3vecB_array vgB=SO3vecB_array::zeros_like(v);
  SO3vecB_array wgB=SO3vecB_array::gaussian_like(w);

#ifdef _WITH_CUDA
  SO3vecB_array ugc=ug.to_device(1);
  SO3vecB_array vgc=vg.to_device(1);
  SO3vecB_array wgc=wg.to_device(1);
  SO3vecB_array ugcB=ug.to_device(1);
  SO3vecB_array vgcB=vg.to_device(1);
  SO3vecB_array wgcB=wg.to_device(1);
#endif


  cout<<"----------- back0 -----------------------"<<endl;

  ug.add_DiagCGproduct_back0(wg,v);
  printl("ug",ug);
  ugB.add_DiagCGproductB_back0(wg,v);
  printl("ugB",ugB);

#ifdef _WITH_CUDA
  ugc.add_DiagCGproduct_back0(wgc,vc);
  printl("ugc",ugc);
  ugcB.add_DiagCGproductB_back0(wgc,vc);
  printl("ugcB",ugcB);
#endif


  cout<<"----------- back1 -----------------------"<<endl;

  vg.add_DiagCGproduct_back1(wg,u);
  printl("vg",vg);
  vgB.add_DiagCGproductB_back1(wg,u);
  printl("vgB",vgB);

#ifdef _WITH_CUDA
  vgc.add_DiagCGproduct_back1(wgc,uc);
  printl("vgc",vgc);
  vgcB.add_DiagCGproductB_back1(wgc,uc);
  printl("vgcB",vgcB);
#endif


  
  cout<<endl; 

}
