#include "GElib_base.cpp"
//#include "GElibSession.hpp"

#include "SO3part.hpp"
#include "CGproduct.hpp"
#include "DiagCGproduct.hpp"
#include "LatexDoc.hpp"
#include "LatexTable.hpp"
#include "TikzPlot.hpp"


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



TensorView<double> benchmark(vector<int> b_values, vector<int> l_values, vector<int> nc_values, const int dev){
  TensorView<double> R({b_values.size(),l_values.size(),l_values.size(),l_values.size(),nc_values.size()},0,0);

  int ntot=0;
  for(int _l1=0; _l1<l_values.size(); _l1++){
    int l1=l_values[_l1];
    for(int _l2=0; _l2<l_values.size(); _l2++){
      int l2=l_values[_l2];
      for(int _l=0; _l<l_values.size(); _l++){
	int l=l_values[_l];
	if(l>=abs(l1-l2) && l<=l1+l2) ntot++;
      }
    }
  }
  ntot*=b_values.size()*nc_values.size();

  int c=0;
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

	    size_t n_ops=((size_t)4)*count_triples(l1,l2,l)*nc*nc*b;
	    int niter=pow(10,min(max(1,6-(int)log10(n_ops)),5));
	    cout<<c<<"/"<<ntot<<" ";
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
	    
	    R.set(_b,_l1,_l2,_l,_nc,speed);
	    cout<<"Time: "<<time<<" ms; ";
	    cout<<"Speed: "<<speed/1000<<" Mflops ";
	    cout<<endl;
	    c++;

	  }
	}
	
      }
    }

  }

  return R;
}

int main(int argc, char** argv){
  int dev=0;
  if(argc>1) dev=stoi(argv[1]);

  vector<int> l_val({2,3,5,7,10,12});
  vector<int> nc_val({4,8,16,32,64});
  vector<int> b_val({1,8,32});

  auto speed=benchmark(b_val,l_val,nc_val,dev);
  TensorView<int> flops0=speed.map<int>([](const Gindex& dummy, const double& x){return (int)(x/1000);});
  

  LatexDoc doc;
  doc<<"\\section*{CG-product}\n";
  
  for(int _b=0; _b<b_val.size(); _b++){

    doc<<"\\subsection*{Batch size "<<b_val[_b]<<"}\n";

    TikzPlot plot(nc_val,flops0.slice(0,_b).slice(1,3).slice(1,3).transp());
    doc<<plot.latex()<<"\\\\ \\\\ \nk";

    for(int lix: vector<int>({1,3,5})){
      if(lix>=l_val.size()) continue;
      for(int ncix: vector<int>({1,3,5,7})){
	if(ncix>=nc_val.size()) continue;

	doc<<"\\subsubsection*{"<<"$\\ell$="<<l_val[lix]<<"~ $n_c$="<<nc_val[ncix]<<"}";
	//doc<<"$\\ell$="<<l_val[lix]<<"\\\\ \n";
	//doc<<"$n_c$="<<nc_val[ncix]<<"\\\\ \n";
	LatexTable<int> table(l_val,l_val,flops0.slice(0,_b).slice(2,lix).slice(2,ncix));
	doc<<table.latex()<<"\\\\ \n";
	doc<<"\\\\ \\\\ \\\\ \n";
      }
    }
  }

  doc.compile("results");

}
