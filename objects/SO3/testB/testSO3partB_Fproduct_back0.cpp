#include "GElib_base.cpp"
#include "SO3vecB.hpp"
//#include "SO3Fvec.hpp"
#include "GElibSession.hpp"


using namespace cnine;
using namespace GElib;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  int b=1;
  int l1=1;
  int l2=1;
  int l=1;

  SO3partB u=SO3partB::Fzero(b,l1);
  SO3partB v=SO3partB::Fgaussian(b,l2);
  SO3partB w=SO3partB::Fgaussian(b,l);
  u.add_Fproduct_back0(w,v);
  cout<<u<<endl;

#ifdef _WITH_CUDA
  SO3partB ug=SO3partB::Fzero(b,l1,1);
  SO3partB vg=v.to_device(1);
  SO3partB wg=w.to_device(1);
  ug.add_Fproduct_back0(wg,vg);
  cout<<ug<<endl;
#endif 

}

