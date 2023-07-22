#include "GElib_base.cpp"
#include "GElibSession.hpp"
#include "DeltaFactor.hpp"

namespace cnine{
  Primes primes;
  FFactorial ffactorial;
  DeltaFactor delta_factor;
}

#include "SO3.hpp"
#include "GprodSpace.hpp"

namespace GElib{
  SO3CouplingMatrices SO3::coupling_matrices;
  GprodSpaceBank<SO3> SO3::product_space_bank;
}

using namespace cnine;
using namespace GElib;

typedef GprodSpace<SO3> SO3space;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  SO3space V1(1);
  SO3space W=V1*V1*V1;
  cout<<W<<endl;


  SO3space W2=W.FmoveL();
  cout<<W2<<endl;

  SO3space W3=W2.FmoveR();
  cout<<W3<<endl;

  cout<<W2.is_isomorphic(W)<<endl;
  cout<<V1.is_isomorphic(W)<<endl;
}
