#include "GElib_base.cpp"
#include "GElibSession.hpp"
#include "DeltaFactor.hpp"

namespace cnine{
  Primes primes;
  FFactorial ffactorial;
  DeltaFactor delta_factor;
}

#include "SO3.hpp"
namespace GElib{
  SO3CouplingMatrices SO3::coupling_matrices;
  GprodSpaceBank<SO3> SO3::product_space_bank;
}

#include "GprodSpace.hpp"

using namespace cnine;
using namespace GElib;

typedef GprodSpace<SO3> SO3basis;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  SO3basis V1(1);
  cout<<V1<<endl;

  SO3basis W=V1*V1*V1*V1;
  cout<<W<<endl;

  cout<<endl;
}
