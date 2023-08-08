#include "Cnine_base.cpp"

#include "CombinatorialBank.hpp"
#include "SnBank.hpp"
namespace Snob2{
  CombinatorialBank* _combibank=new CombinatorialBank();
  SnBank* _snbank=new SnBank();
}

#include "GElib_base.cpp"
#include "GElibSession.hpp"
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

  SO3basis A=(V1*V1*V1);
  for(auto& p:A.obj->isotypics)
    p.second->Snisotypics();
  cout<<endl<<endl<<endl;

  SO3basis B=(V1*V1*V1*V1);
  for(auto& p:B.obj->isotypics)
  p.second->Snisotypics();
  cout<<endl<<endl<<endl;

  SO3basis C=(V1*V1*V1*V1*V1);
  for(auto& p:C.obj->isotypics)
  p.second->Snisotypics();
  cout<<endl<<endl<<endl;

}

