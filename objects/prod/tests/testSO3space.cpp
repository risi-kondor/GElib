#include "GElib_base.cpp"
#include "GElibSession.hpp"
#include "SO3.hpp"

using namespace cnine;
using namespace GElib;


#include "GSnSpaceBank.hpp"
GSnSpaceBank<SO3> GSnSpace_bank;
#include "GSnSpace.hpp"

typedef GSnSpace<SO3> SO3space;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  SO3space V1(1);
  cout<<V1.repr()<<endl;

  SO3space W=V1*V1*V1;
  cout<<W.repr()<<endl;

  cout<<endl;
}
