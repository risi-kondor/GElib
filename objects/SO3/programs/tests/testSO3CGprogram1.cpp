#include "GElib_base.cpp"
#include "SO3vecB.hpp"
#include "GElibSession.hpp"

#include "SO3CGprogramBank.hpp"

GElib::SO3CGprogramBank SO3_CGprogram_bank;

#include "SO3GenCGproducts.hpp"


using namespace cnine;
using namespace GElib;

typedef CtensorB ctensor;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  SO3type tau={2,2};

  SO3vecB u=SO3vecB::gaussian(1,tau);
  SO3vecB v=SO3vecB::gaussian(1,tau);
  printl("u",u)<<endl;
  printl("v",v)<<endl;

  SO3vecB w=CGproduct({u,u,u}); 
  cout<<w<<endl;

}
