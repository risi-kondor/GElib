#ifndef _CnineIntegerPartition
#define _CnineIntegerPartition

#include "Cnine_base.hpp"


namespace cnine{

  class IntegerPartition{
  public: 

    int k;
    int* p=nullptr;

    
  public:

    IntegerPartition(): k(0){
      p=nullptr;
    }

    IntegerPartition(const int _k, const cnine::fill_raw& dummy): k(_k){
      p=new int[k];
    }

    IntegerPartition(const int n, const cnine::fill_identity& dummy): k(1){
      p=new int[1]; 
      p[0]=n;
    }

    IntegerPartition(const int _k, const cnine::fill_zero& dummy): k(_k){
      p=new int[k]; 
      for(int i=0; i<k; i++) p[i]=0;
    }

    IntegerPartition(const initializer_list<int> list): 
      IntegerPartition(list.size(),cnine::fill_raw()){
      int i=0; for(auto v:list) p[i++]=v;
    }	

    IntegerPartition(const vector<int> list): 
      IntegerPartition(list.size(),cnine::fill_raw()){
      int i=0; for(auto v:list) p[i++]=v;
    }	


  public: // copying 

    IntegerPartition(const IntegerPartition& x):k(x.k){
      p=new int[k]; 
      for(int i=0; i<k; i++) p[i]=x.p[i];
    }

    IntegerPartition& operator=(const IntegerPartition& x){
      k=x.k; if(p) delete[] p; p=new int[k]; 
      for(int i=0; i<k; i++) p[i]=x.p[i]; 
      return *this;
    }
  
    IntegerPartition(IntegerPartition&& x): k(x.k) {
      p=x.p; x.p=nullptr;}
  
    IntegerPartition& operator=(IntegerPartition&& x){
      if (this!=&x) {k=x.k; if(p) delete[] p; p=x.p; x.p=nullptr;}
      return *this;
    }
  
    ~IntegerPartition(){if(p) delete[] p;}


  private:

    int factorial(int x) const{
      return x==0 ? 1 : x*=factorial(x-1);
    }


  public: // Access

    int height() const{
      return k;
    }

    int getn() const{
      int n=0; 
      for(int i=0; i<k; i++) n+=p[i]; 
      return n;
    }

    int& operator[](const int r){
      return p[r];
    }

    int operator[](const int r) const{
      return p[r];
    }

    int& operator()(const int r){
      return p[r];
    }

    int operator()(const int r) const{
      return p[r];
    }

    void set(const int r, const int x){
      p[r]=x;
    }


  public:

    int hooklength() const {
      int res = factorial(getn());
      for(int r=1; r<=k; r++){
	for(int c=1; c<=p[r-1]; c++){
	  int right = p[r-1] - c;
	  int below = 0;
	  for(int i=r+1; i<=k; i++)
	    below += (p[i-1] >=c ? 1 : 0);
	  res /= (right+below+1);
	}
      }
      return res;
    };

    bool extendable(const int i) const{
      if(i==k) return true;
      if(i==0) return true;
      if(p[i-1]>p[i]) return true;
      return false;
    }

    bool shortenable(const int i) const{
      if(i==k-1) return true;
      if(p[i+1]<p[i]) return true;
      return false;
    }

    int content_of_difference(const IntegerPartition& mu) const{
      CNINE_ASSRT(getn()==mu.getn()+1);
      CNINE_ASSRT(height()>=mu.height());
      int rowix=mu.height();
      for(int i=0; i<mu.height(); i++)
	if(mu[i]==p[i]-1){
	  rowix=i;
	  break;
	}
      return p[rowix]-1-rowix;
    }

    IntegerPartition& add(const int r, const int m=1){
      if(r<k){p[r]+=m; return *this;}
      int* newp=new int[k+1]; 
      for(int i=0; i<k; i++) newp[i]=p[i]; newp[k]=m;
      k++; delete[] p; p=newp; 
      return *this; 
    }

    IntegerPartition& remove(const int r, const int m=1){
      p[r]-=m;
      if(p[k-1]==0){
	int* newp=new int[k-1]; 
	for(int i=0; i<k-1; i++) newp[i]=p[i]; 
	k--; delete[] p; p=newp;
      }
      return* this; 
    }

    void for_each_sub(std::function<void(const IntegerPartition&)> fun) const{
      return foreach_sub(fun);
    }

    void foreach_sub(std::function<void(const IntegerPartition&)> fun) const{
      IntegerPartition lambda(*this);
      int k=lambda.k;
      assert(lambda.k>0);

      lambda.p[k-1]--;
      if(lambda.p[k-1]==0){
	lambda.k--;
	fun(lambda);
	lambda.k++;
      }else{
	fun(lambda);
      }
      lambda.p[k-1]++;

      for(int i=k-2; i>=0; i--){
	if(lambda.p[i+1]<lambda.p[i]){
	  lambda.p[i]--;
	  fun(lambda);
	  lambda.p[i]++;
	}
      }

    }

    bool operator==(const IntegerPartition& x) const{
      int i=0; for(; i<k && i<x.k; i++) if (p[i]!=x.p[i]) return false;
      for(;i<k; i++) if (p[i]!=0) return false;
      for(;i<x.k; i++) if(x.p[i]!=0) return false;
      return true;
    }

    bool operator<(const IntegerPartition&  y) const{
      CNINE_ASSERT(getn()==y.getn(),"Comparing partition of n with partition of m<>n.");
      for(int i=0; i<std::min(k,y.k); i++)
	if(p[i]>y.p[i]) return true;
	else if(p[i]<y.p[i]) return false; 
      //if(k<y.k) return true;
      return false;
    }

  public:


    vector<IntegerPartition> parents() const{
      vector<IntegerPartition> R;
      for_each_sub([&](const IntegerPartition& mu){
	  R.push_back(mu);});
      return R;
    }


  public: // I/O

    string str(const string indent="") const{
      ostringstream oss;
      oss<<"["; 
      for(int i=0; i<k; i++){
	oss<<p[i];
	if(i<k-1) oss<<",";
      }
      oss<<"]"; 
      return oss.str();
    }
     
    
    friend ostream& operator<<(ostream& stream, const IntegerPartition& x){
      stream<<x.str(); return stream;
    }


  };

}

namespace std{
  template<>
  struct hash<cnine::IntegerPartition>{
  public:
    size_t operator()(const cnine::IntegerPartition& x) const{
      size_t r=1;
      for(int i=0; i<x.k; i++) r=(r<<1)^hash<int>()(x.p[i]);
      return r;
    }
  };
}


/*
namespace std{
  template<>
  struct hash<std::pair<Snob2::IntegerPartition,Snob2::IntegerPartition> >{
  public:
    size_t operator()(const pair<Snob2::IntegerPartition,Snob2::IntegerPartition>& lambdas) const{
      return (hash<Snob2::IntegerPartition>()(lambdas.first)<<1)^
	(hash<Snob2::IntegerPartition>()(lambdas.second));
    }
  };
}


namespace std{
  template<>
  struct hash<std::tuple<Snob2::IntegerPartition,Snob2::IntegerPartition,Snob2::IntegerPartition> >{
  public:
    size_t operator()(const tuple<Snob2::IntegerPartition,Snob2::IntegerPartition,Snob2::IntegerPartition>& lambdas) const{
      return ((hash<Snob2::IntegerPartition>()(std::get<0>(lambdas))<<1)^
	(hash<Snob2::IntegerPartition>()(std::get<1>(lambdas))<<1))^
	(hash<Snob2::IntegerPartition>()(std::get<2>(lambdas)));
    }
  };
}
*/

#endif
