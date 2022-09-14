#include "GElib_base.cpp"
#include "SO3vecB.hpp"
#include "GElibSession.hpp"
#include "SO3vecB.hpp"
#include "SO3weights.hpp"


using namespace cnine;
using namespace GElib;

typedef CtensorB ctensor;


int main(int argc, char** argv){
  GElibSession session(4);
  cout<<endl;

  SO3type tau1({1,2,1});
  SO3type tau2({2,2,3});

  SO3weights W=SO3weights::zero(tau1,tau2);
  cout<<W<<endl;

  SO3vecB u=SO3vecB::gaussian(1,tau1);
  SO3vecB v=u*W;

  cout<<v<<endl;


}
