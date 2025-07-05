#include "GElib_base.cpp"
#include "GElibSession.hpp"
#include "O3vec.hpp"
#include "CGproduct.hpp"
#include "DiagCGproduct.hpp"

using namespace cnine;
using namespace GElib;

int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  O3vec<float> u(ttype=O3type({{{2,1},1},{{3,-1},1}}),filltype=4);
  O3vec<float> v(ttype=O3type({{{2,1},1},{{3,-1},1}}),filltype=4);
  print(v);

  auto w=CGproduct(u,v);
  print(w);

  auto z=DiagCGproduct(u,v);
  print(z);

}
