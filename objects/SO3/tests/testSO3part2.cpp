//#define DEBUG_ENGINE_FLAG
//#define CENGINE_ECHO_QUEUE

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

  const SO3part u(2,2,fill::gaussian);
  printl("u",u)<<endl;

  cout<<u.get_value(0,1)<<endl; 
  cout<<u(0,1)<<endl;
  cout<<endl<<endl;


  SO3part v(2,2,fill::gaussian);
  printl("v",u)<<endl;

  cout<<v.get_value(0,1)<<endl; 
  cout<<v(0,1)<<endl;


}
