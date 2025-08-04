#include "GElib_base.cpp"
//#include "GElibSession.hpp"

#include "SO3part.hpp"
#include "CGproduct.hpp"
#include "DiagCGproduct.hpp"

using namespace cnine;
using namespace GElib;


class Timer{
  public:

  chrono::time_point<chrono::system_clock> t0;

  Timer(){
    t0=chrono::system_clock::now();
  }

  double time(){
    return chrono::duration<double,std::milli>(chrono::system_clock::now()-t0).count();
  }
};


int count_triples(int l1, int l2, int l){
  int r=0;
  for(int m1=-l1; m1<=l1; m1++)
    for(int m2=-l2; m2<=l2; m2++)
      if(abs(m1+m2)<l) r++;
  return r;
}


int main(int argc, char** argv){
  int dev=0;
  if(argc>1) dev=stoi(argv[1]);

  //int niter=10;
  vector<int> l_values({1,2,3,5,7,12});
  vector<int> nc_values({4,8,16,32,64});
  vector<int> b_values({1,8});

  TensorView<double> flops1({b_values.size(),l_values.size(),l_values.size(),l_values.size(),nc_values.size()},0,0);

  for(int _b=0; _b<b_values.size(); _b++){
    int b=b_values[_b];
    for(int _l1=0; _l1<l_values.size(); _l1++){
      int l1=l_values[_l1];
      for(int _l2=0; _l2<l_values.size(); _l2++){
	int l2=l_values[_l2];
	for(int _l=0; _l<l_values.size(); _l++){
	  int l=l_values[_l];
	  if(l<abs(l1-l2) || l>l1+l2) continue;
	  for(int _nc=0; _nc<nc_values.size(); _nc++){
	    int nc=nc_values[_nc];

	    size_t n_ops=((size_t)4)*count_triples(l1,l2,l)*nc*nc*b*b;
	    int niter=pow(10,min(max(1,6-(int)log10(n_ops)),5));
	    cout<<"Running "<<niter<<" CG-products with ";
	    cout<<"b="<<b<<" ";
	    cout<<"l1="<<l1<<" ";
	    cout<<"l2="<<l2<<" ";
	    cout<<"l="<<l<<" ";
	    cout<<"nc="<<nc<<" ";
	    cout<<"...   ";//<<endl;

	    SO3part<float> P0(irrep=l1,batch=b,channels=nc,filltype=4,device=dev);
	    SO3part<float> P1(irrep=l2,batch=b,channels=nc,filltype=4,device=dev);

	    Timer timer;
	    for(int i=0; i<niter; i++)
	      auto A=CGproduct(P0,P1,l);
	    auto time=timer.time();
	    double speed=n_ops*niter/time;
	    
	    flops1.set(_b,_l1,_l2,_l,_nc,speed);
	    cout<<"Time: "<<time<<" ms; ";
	    cout<<"Speed: "<<speed/1000<<" Mflops ";
	    cout<<endl;
	    
	  }
	}
	
      }
    }

  }

}
