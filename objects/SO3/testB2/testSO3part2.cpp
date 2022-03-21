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

  const SO3part u(1,2,2,fill::gaussian);
  printl("u",u)<<endl;

}
