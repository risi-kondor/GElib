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
  int bsize=1;
  SO3type tau({2,2});

  SO3vecB u=SO3vecB::gaussian(1,tau);
  SO3vecB v=SO3vecB::gaussian(1,tau);

  SO3vecB w=u.CGproduct(v,2);
  cout<<w<<endl;

  SO3vecB wb=u.BlockedCGproduct(v,bsize,2);
  cout<<wb<<endl;

  cout<<endl; 

#ifdef _WITH_CUDA
  SO3vecB uc=u.to_device(1);
  SO3vecB vc=v.to_device(1);
  SO3vecB wc=uc.BlockedCGproduct(vc,bsize,2);
  cout<<wc<<endl;
#endif 
  
  SO3vecB ug=SO3vecB::zeros_like(u);
  SO3vecB vg=SO3vecB::zeros_like(v);
  SO3vecB wg=SO3vecB::gaussian_like(wb);

#ifdef _WITH_CUDA
  SO3vecB ugc=ug.to_device(1);
  SO3vecB vgc=vg.to_device(1);
  SO3vecB wgc=wg.to_device(1);
#endif

  cout<<"----------- back0 -----------------------"<<endl;

  ug.add_BlockedCGproduct_back0(wg,v,bsize);
  printl("ug",ug);

#ifdef _WITH_CUDA
  ugc.add_BlockedCGproduct_back0(wgc,vc,bsize);
  printl("ugc",ugc);
#endif

  cout<<"----------- back1 -----------------------"<<endl;

  vg.add_BlockedCGproduct_back1(wg,u,bsize);
  printl("vg",vg);

#ifdef _WITH_CUDA
  vgc.add_BlockedCGproduct_back1(wgc,uc,bsize);
  printl("vgc",vgc);
#endif 

}

