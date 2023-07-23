#include "GElib_base.cpp"
#include "GElibSession.hpp"
#include "GprodBasis.hpp"
#include "SO3.hpp"

namespace GElib{
  SO3CouplingMatrices SO3::coupling_matrices;
  GprodSpaceBank<SO3> SO3::product_space_bank;
}


using namespace cnine;
using namespace GElib;

typedef GprodBasis<SO3> SO3basis;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  SO3basis V1(1);
  SO3basis W=V1*V1*V1;
  cout<<W<<endl;


  SO3basis W2=W.shift_left();
  cout<<W2<<endl;

  //auto A=W.coupling(W2);
  //cout<<A<<endl;

}
