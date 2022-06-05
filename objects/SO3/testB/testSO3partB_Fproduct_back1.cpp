#include "GElib_base.cpp"
#include "SO3vecB.hpp"
// #include "SO3Fvec.hpp"
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

  SO3partB u=SO3partB::Fgaussian(b,l1);
  SO3partB v=SO3partB::Fzero(b,l2);
  SO3partB w=SO3partB::Fgaussian(b,l);
  v.add_Fproduct_back1(w,u);
  cout<<v<<endl;

#ifdef _WITH_CUDA
  SO3partB ug=u.to_device(1);
  SO3partB vg=SO3partB::Fzero(b,l2,1);
  SO3partB wg=w.to_device(1);
  vg.add_Fproduct_back1(wg,ug);
  cout<<vg<<endl;
#endif 

}

