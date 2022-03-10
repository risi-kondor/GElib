#include "GElib_base.cpp"
#include "SO3vecB.hpp"
#include "GElibSession.hpp"


using namespace cnine;
using namespace GElib;

typedef CtensorB ctensor;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  int b=2;
  SO3type tau({2,2,2});
  int maxl=2;
  float eps=10e-6;

  /*
  SO3partB x=SO3partB::gaussian(b,1,1);
  SO3partB y=SO3partB::gaussian(b,1,1);

  SO3partB z=x.CGproduct(y,2);
  cout<<z<<endl;
  
  SO3partB zg=SO3partB::zeros_like(z);
  zg.view().inc(0,2,0,1.0);
  SO3partB xg=SO3partB::zeros_like(x);
  xg.add_CGproduct_back0(zg,y);
  cout<<xg<<endl;

  x.view().inc(0,1,0,1.0);
  cout<<x.CGproduct(y,2)-z<<endl;
  */

  SO3vecB x=SO3vecB::gaussian(b,tau);
  SO3vecB y=SO3vecB::gaussian(b,tau);

  SO3vecB z=x.CGproduct(y,maxl);
  cout<<z<<endl;

  SO3vecB zg=SO3vecB::zeros_like(z);
  zg.parts[1]->view().inc(1,1,2,std::complex<float>(0,1.0));
  SO3vecB xg=SO3vecB::zeros_like(x);
  xg.add_CGproduct_back0(zg,y);
  cout<<xg<<endl;

  SO3vecB delta=SO3vecB::zero(b,tau);
  x.parts[1]->view().inc(1,0,0,std::complex<float>(0,1.0));
  cout<<x.CGproduct(y,maxl)-z<<endl;
  

}

