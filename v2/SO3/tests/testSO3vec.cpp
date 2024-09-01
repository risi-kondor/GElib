#include "GElib_base.cpp"
//#include "GElibSession.hpp"
#include "SO3vec.hpp"
#include "SO3functions.hpp"

using namespace cnine;
using namespace GElib;

int main(int argc, char** argv){
  //GElibSession session;
  cout<<endl;

  SO3vec<float> v(ttype=SO3type({{2,1},{3,1}}));
  print(v);

  auto w=CGproduct(v,v);
  print(w);

}
