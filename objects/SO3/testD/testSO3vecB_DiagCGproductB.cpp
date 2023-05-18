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
  cout<<wc<<endl;

  SO3vecB_array wcB=uc.DiagCGproductB(vc,2);
  cout<<wcB<<endl;
#endif 
    
  cout<<endl; 

}
