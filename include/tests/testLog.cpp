#include "GElib_base.cpp"
#include "SO3partB.hpp"
#include "GElibSession.hpp"
#include "GElibLog.hpp"
#include "GElibTimer.hpp"

using namespace cnine;
using namespace GElib;

int main(int argc, char** argv){
  GElibSession session(4);
  GElibLog gelib_log;

  {
    LoggedTimer timer("Input from user");
    int t=0;
    cin>>t;
  }
  LoggedTimer t;

}
