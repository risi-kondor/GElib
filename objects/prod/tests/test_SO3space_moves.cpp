#include "GElib_base.cpp"
#include "GElibSession.hpp"
#include "DeltaFactor.hpp"
#include "CGprodBasis.hpp"
#include "SO3.hpp"

namespace GElib{
  SO3CouplingMatrices SO3::coupling_matrices;
  CGprodBasisBank<SO3> SO3::product_space_bank;
}


using namespace cnine;
using namespace GElib;

typedef CGprodBasis<SO3> SO3basis;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  SO3basis V1(1);
  SO3basis W=V1*V1*V1;
  cout<<W<<endl;


  SO3basis W2=W.shift_left();
  cout<<W2<<endl;

  SO3basis W3=W2.shift_right();
  cout<<W3<<endl;

  cout<<W2.is_isomorphic(W)<<endl;
  cout<<V1.is_isomorphic(W)<<endl;
}
