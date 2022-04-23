#include "GElib_base.cpp"
#include "SO3partB.hpp"
#include "GElibSession.hpp"


using namespace cnine;
using namespace GElib;

extern GElib::SO3_CGbank SO3_cgbank;

int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  const int b=1;
  const int n=1;
  const int l=30;

  SO3element R0(fill::uniform);
  SO3element R1(fill::uniform);
  SO3element R(fill::uniform);
  //printl("R",R);
  //CtensorB D(WignerMatrix<float>(l,R));
  //cout<<D<<endl;

  for(int l0=10; l0<=20; l0++)
    for(int l1=10; l1<=20; l1++){
      for(int l=25; l<=l0+l1; l++){
	cout<<l0<<" "<<l1<<" "<<l<<":"<<endl;

      SO3partB u=SO3partB::gaussian(b,l0,n);
      SO3partB v=SO3partB::gaussian(b,l1,n);
      //printl("u",u)<<endl;
      //printl("v",v)<<endl;


      SO3partB w=u.CGproduct(v,l);
      SO3partB wr=w.rotate(R);
      //printl("w.rotate(R)",wr);

      SO3partB uR=u.rotate(R);
      SO3partB vR=v.rotate(R);
      SO3partB wR=uR.CGproduct(vR,l);

      //printl("wR",wR);
      cout<<"diff2="<<sqrt(wr.diff2(wR))/sqrt(wr.norm2())<<endl;
      cout<<endl;
    }
    }

  cout<<endl; 
}
