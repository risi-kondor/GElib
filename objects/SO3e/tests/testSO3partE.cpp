#include "GElib_base.cpp"
#include "GElibSession.hpp"
#include "Ltensor.hpp"

//#include "SO3groupE.hpp"
//#include "GpartE.hpp"
#include "SO3partE.hpp"

using namespace cnine;
using namespace GElib;

typedef GpartE<complex<float> > Gpart;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  int b=3;
  int l=2;
  int n=2;

  auto u=SO3partE<float>::gaussian().l(l).n(n)();
  auto v=SO3partE<float>::gaussian().batch(b).l(l).n(n)();
  cout<<u<<endl;
  cout<<v<<endl;

  auto M=Ltensor<complex<float> >::gaussian().dims({2,3})();
  cout<<M<<endl;
  cout<<u*M<<endl;

  auto w=CGproduct(u,u,2);
  cout<<w<<endl;


}
