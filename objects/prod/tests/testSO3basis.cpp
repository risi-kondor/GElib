#include "GElib_base.cpp"
#include "GElibSession.hpp"
#include "CGprodBasis.hpp"
#include "SO3.hpp"

namespace GElib{
  SO3CouplingMatrices SO3::coupling_matrices;
  CGprodBasisBank<SO3> SO3::product_space_bank;
  template<> int CGprodBasisObj<SO3>::indnt=0;
}

using namespace cnine;
using namespace GElib;

typedef CGprodBasis<SO3> SO3basis;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  SO3basis V1(1);
  cout<<V1<<endl;

  SO3basis W=V1*V1*V1*V1;
  cout<<W<<endl;

  SO3basis E=(V1*V1*V1);
  //SO3basis E=(V1*(V1*V1))*V1;
  //SO3basis E=(V1*(V1*V1))*(V1*V1*V1);
  cout<<E<<endl;
  //cout<<E.shift_left()<<endl;
  cout<<E.standard_form()<<endl;
  cout<<E.standardizing_map()<<endl;

  auto U=E.standardizing_map();
  for(auto& p:U.maps){
    auto& A=p.second;
    //cout<<A<<endl;
    cout<<cnine::transp(A)*A<<endl;
  }

  cout<<E.swap_map()<<endl;
  cout<<E.transpose_last_map()<<endl;

  cout<<"---------"<<endl;

  for(auto& p:E.obj->isotypics)
    cout<<p.second->Sn_basis()<<endl;
    //cout<<*p.second<<endl;

  cout<<endl;
}
