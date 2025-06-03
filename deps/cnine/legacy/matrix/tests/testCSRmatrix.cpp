#include "Cnine_base.cpp"
#include "RtensorA.hpp"
#include "CnineSession.hpp"
#include "CSRmatrix.hpp"

using namespace cnine;

typedef RtensorA rtensor; 


int main(int argc, char** argv){

  cnine_session session;
  cout<<endl;

  int n=5;

  RtensorA A=RtensorA::gaussian({5,5});
  for(int i=0; i<n; i++)
    for(int j=0; j<n; j++)
      if(A(i,j)<0.1) A.set(i,j,0.0);
  cout<<A<<endl;

  CSRmatrix<float> B(A);
  cout<<B<<endl;

  CSRmatrix<float> C=B.transp();
  cout<<C<<endl;


}
