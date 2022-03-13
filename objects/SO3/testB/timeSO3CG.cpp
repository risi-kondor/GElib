#include "GElib_base.cpp"
#include "SO3vecB.hpp"
#include "GElibSession.hpp"
#include <chrono>


using namespace cnine;
using namespace GElib;

typedef CtensorB ctensor;


class Timer{
  public:

  chrono::time_point<chrono::system_clock> t0;

  Timer(){
    t0=chrono::system_clock::now();
  }

  double time(){
      return chrono::duration<double>(chrono::system_clock::now()-t0).count();
  }
};


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  int b=512;
  int l1=3;
  int l2=3;
  int l=3;
  int maxl=15;
  int n1=32;
  int n2=32;
  int niter=1;

#ifdef _WITH_CUDA
  SO3partB u0=SO3partB::gaussian(1,l1,1,1);
  SO3partB v0=SO3partB::gaussian(1,l2,1,1);
  SO3partB w0=u0.CGproduct(v0,l);
#endif

vector<SO3partB*> U;
for(int _l=0; _l<maxl; _l++)
  U.push_back(new SO3partB(SO3partB::gaussian(b,_l,n1)));
vector<SO3partB*> V;
for(int _l=0; _l<maxl; _l++)
  V.push_back(new SO3partB(SO3partB::gaussian(b,_l,n2)));

#ifdef _WITH_CUDA
vector<SO3partB*> Ug;
for(int _l=0; _l<maxl; _l++)
  Ug.push_back(new SO3partB(SO3partB::gaussian(b,_l,n1,1)));
vector<SO3partB*> Vg;
for(int _l=0; _l<maxl; _l++)
  Vg.push_back(new SO3partB(SO3partB::gaussian(b,_l,n2,1)));
#endif 

if(true){
  cout<<"Starting CPU"<<endl;
  for(int _l=0; _l<maxl; _l++){
    Timer T;
    for(int i=0; i<niter; i++){
      SO3partB w=(*U[_l]).CGproduct(*V[_l],_l);
      //cout<<w<<endl;
    }
    cout<<_l<<": "<<T.time()<<endl;
  }
  cout<<"."<<endl;
}

#ifdef _WITH_CUDA
 cout<<"Starting GPU"<<endl;
  for(int _l=0; _l<maxl; _l++){
    Timer T;
    for(int i=0; i<niter; i++){
      SO3partB w=(*Ug[_l]).CGproduct(*Vg[_l],_l);
    }
    cout<<_l<<": "<<T.time()<<endl;
  }
  cout<<"."<<endl;
#endif 

  cout<<endl; 

}

