#include "GElib_base.cpp"
#include "GElibSession.hpp"
#include "GprodBasis.hpp"
#include "SO3.hpp"

namespace GElib{
  SO3CouplingMatrices SO3::coupling_matrices;
  GprodSpaceBank<SO3> SO3::product_space_bank;
  template<> int CGprodBasisObj<SO3>::indnt=0;
}

using namespace cnine;
using namespace GElib;

typedef GprodBasis<SO3> SO3basis;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  SO3basis V1(1);
  cout<<V1<<endl;

  SO3basis W=V1*V1*V1*V1;
  cout<<W<<endl;

  SO3basis E=(V1*(V1*V1))*(V1*V1*V1);
  cout<<E<<endl;
  cout<<E.shift_left()<<endl;
  cout<<E.standard_form()<<endl;

  //E.obj->standardize();
  cout<<E.standardizing_map()<<endl;

  cout<<endl;
}
