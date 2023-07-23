#include "GElib_base.cpp"
#include "GElibSession.hpp"
#include "SO3.hpp"

namespace GElib{
  SO3CouplingMatrices SO3::coupling_matrices;
  GprodSpaceBank<SO3> SO3::product_space_bank;
}

using namespace cnine;
using namespace GElib;




int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;


  auto M=SO3::coupling(1,1,1,1);
  cout<<M<<endl;

}
