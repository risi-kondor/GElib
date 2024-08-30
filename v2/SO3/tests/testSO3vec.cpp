#include "GElib_base.cpp"
//#include "GElibSession.hpp"
#include "SO3vec.hpp"

using namespace cnine;
using namespace GElib;

int main(int argc, char** argv){
  //GElibSession session;
  cout<<endl;

  SO3vec<float> v(batch=2,ttype=SO3type({{2,3},{3,1}}));
  print(v);

}
