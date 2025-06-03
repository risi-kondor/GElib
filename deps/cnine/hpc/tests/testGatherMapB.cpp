#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "GatherMapB.hpp"
#include "GatherRows.hpp"
#include "Ltensor.hpp"

using namespace cnine;

int main(int argc, char** argv){

  cnine_session session;

  GatherMapB g=GatherMapB::random(10,10,0.2);
  cout<<g<<endl;
  cout<<g.sort()<<endl;
  cout<<g.inv()<<endl;
  cout<<g<<endl;

  Ltensor<int> A(dims(10,10));
  uniform_int_distribution<int> distr(0,4);
  for(int i=0; i<10; i++)
    for(int j=0; j<10; j++)
      A.set(i,j,distr(rndGen));
  cout<<A<<endl;

  Ltensor<int> B=GatherRows()(A,g);
  cout<<B<<endl;

  Ltensor<int> U(dims(5,2));
  U(0,0)=1;
  U(0,1)=3;
  U(1,0)=4;
  U(1,1)=2;
  U(2,0)=0;
  U(2,1)=8;
  U(3,0)=9;
  U(3,1)=4;
  U(4,0)=3;
  U(4,1)=3;
  cout<<U<<endl;

  GatherMapB g2(U);
  cout<<g2<<endl;

}


