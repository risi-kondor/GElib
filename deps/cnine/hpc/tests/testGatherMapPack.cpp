#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "GatherMapB.hpp"
#include "GatherMapPack.hpp"
#include "GatherRows.hpp"
#include "Ltensor.hpp"

using namespace cnine;

int main(int argc, char** argv){

  cnine_session session;

  GatherMapB g=GatherMapB::random(10,10,0.2);
  cout<<g<<endl;

  Ltensor<int> a(dims(10,10));
  uniform_int_distribution<int> distr(0,4);
  for(int i=0; i<10; i++)
    for(int j=0; j<10; j++)
      a.set(i,j,distr(rndGen));
  //cout<<a<<endl;

  auto A=Ltensor<int>::stack(0,{a,a,a});
  cout<<A<<endl;

  auto gp=shared_ptr<GatherMapB>(new GatherMapB(g));
  GatherMapPack G(gp);
  G.push_back(gp);
  G.push_back(gp);
  cout<<G<<endl;

  Ltensor<int> B=GatherRows()(A,G);
  cout<<B<<endl;


  

}


