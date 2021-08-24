#include "GElib_base.cpp"
#include "SO3partArray.hpp"
#include "GElibSession.hpp"

#include "CGprod.hpp"

using namespace cnine;
using namespace GElib;

typedef CscalarObj cscalar;
typedef CtensorObj ctensor;


int main(int argc, char** argv){
  GElibSession session;

  const int N=10;
  const Gdims adims({N,N});
  const SO3type tau0({1,1});

  const CtensorArray coords(dims(N),dims(3),fill::gaussian);
  const Ctensor dist(adims,[&](const int i, const int j){
      return 1.0/norm2(coords.get_cell(i)-coords.get_cell(j)).get_value();});

  SO3vecArray L0(adims,tau,fill::raw);
  for(int i=0; i<N; i++)
    for(int j=0; j<N; j++)
      L0.cell(i,j)=SO3vec(tau0,coords.get_cell(i)-coords.get_cell(j));

  auto M1=CGproduct(outer(L0,L0))*scatter(dist);
  auto L1=M1.reduce(1)*W1;
      

}
