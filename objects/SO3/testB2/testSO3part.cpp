#include "GElib_base.cpp"
#include "SO3part.hpp"
#include "GElibSession.hpp"


using namespace cnine;
using namespace GElib;

typedef CscalarObj cscalar;
typedef CtensorObj ctensor;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;


  SO3part u=SO3part::gaussian(2,2);
  SO3part v=SO3part::gaussian(2,2);
  printl("u",u)<<endl;
  printl("v",v)<<endl;


  cout<<"CGproduct(u,v,2)="<<endl<<CGproduct(u,v,2)<<endl<<endl;;

  cout<<endl; 
}
