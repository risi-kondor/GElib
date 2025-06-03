
// ---- Helper functions -----------------------------------------------------------------------------------


namespace cnine{

  template<typename TYPE>
  inline TYPE& unconst(const TYPE& x){
    return const_cast<TYPE&>(x);
  }

  inline int roundup(const int x, const int s){
    return ((x-1)/s+1)*s;
  }

  inline int roundup(const int x){
    return ((x-1)/32+1)*32;
  }

  template<typename TYPE>
  inline TYPE ifthen(const bool p, const TYPE& x, const TYPE& y){
    if(p) return x; else return y;
  }

  template<typename TYPE>
  inline TYPE bump(TYPE& x, TYPE y){
    if(y>x) x=y;
    return x;
  }

  template<typename TYPE>
  inline void fastadd(const TYPE* source, TYPE* dest, const int n){
    for(int i=0; i<n; i++)
      *(dest+i)+=*(source+i);
  }

  template<typename TYPE>
  void stdadd(const TYPE* beg, const TYPE* end, TYPE* dest){
    const int n=end-beg; 
    for(int i=0; i<n; i++)
      dest[i]+=beg[i];
  }

  template<typename TYPE>
  void stdadd(const TYPE* beg, const TYPE* end, TYPE* dest, TYPE c){
    const int n=end-beg; 
    for(int i=0; i<n; i++)
      dest[i]+=c*beg[i];
  }

  template<typename TYPE>
  void stdsub(const TYPE* beg, const TYPE* end, TYPE* dest){
    const int n=end-beg; 
    for(int i=0; i<n; i++)
      dest[i]-=beg[i];
  }

  template<typename TYPE>
  inline std::vector<TYPE> permute(const std::vector<TYPE> x, const std::vector<TYPE> pi){
    int n=pi.size();
    CNINE_ASSRT(x.size()==n);
    std::vector<TYPE> R(n);
    for(int i=0; i<n; i++)
      R[i]=x[pi[i]];
    return R;
  }

  template<typename TYPE1, typename TYPE2>
  inline std::vector<TYPE1> convert(std::vector<TYPE2>& x){
    std::vector<TYPE1> R(x.size());
    for(int i=0; i<x.size(); i++)
      R[i]=TYPE1(x[i]);
    return R;
  }

  template<typename TYPE1, typename TYPE2>
  inline std::vector<TYPE2> mapcar(const std::vector<TYPE1>& v, 
    const std::function<TYPE2(const TYPE1&)> lambda){
    std::vector<TYPE2> R;
    for(auto& p: v)
      R.push_back(lambda(p));
    return R;
  }

  template<typename ARG1, typename ARG2, typename RESULT>
  inline std::vector<RESULT> mapcar(const std::vector<ARG1>& x1, const std::vector<ARG2>& x2, 
    const std::function<RESULT(const ARG1&, const ARG2&)> lambda){
    CNINE_ASSRT(x1.size()==x2.size());
    std::vector<RESULT> R;
    int N=x1.size();
    for(int i=0; i<N; i++)
      R.push_back(lambda(x1[i],x2[i]));
    return R;
  }

  template<typename TYPE>
  inline vector<TYPE> except(const vector<TYPE>& x, const vector<int>& ix){
    int N=x.size();
    int n=ix.size();

    vector<TYPE> R(N-n);
    int last=0;
    auto beg=x.begin(); 
    auto tail=R.begin();
    for(int i=0; i<ix.size(); i++){
      CNINE_ASSRT(ix[i]<N);
      int len=ix[i]-last;
      std::copy(beg,beg+len,tail);
      beg+=len+1;
      tail+=len;
      last=ix[i]+1;
    }
    if(last<N)
      std::copy(beg,x.end(),tail);
    return R;
  }


  // ---- Printing -------------------------------------------------------------------------------------------


  //template<typename TYPE>
  //void print(const TYPE& x){
  //cout<<x.str()<<endl;
  //}

  template<typename TYPE>
  inline ostream& print(const string name, const TYPE& x){
    cout<<name<<"="<<x.str()<<endl;
    return cout; 
  }

  template<typename TYPE>
  inline ostream& printl(const string name, const TYPE& x){
    cout<<name<<"="<<endl<<x.str()<<endl;
    return cout; 
  }

  template<typename TYPE>
  auto show_if_possible(const TYPE& x, int) -> decltype(x.to_print(),std::to_string(1)){
    return x.to_print();
  }

  template<typename TYPE>
  string show_if_possible(const TYPE& x, long){
    return x.str();
  }

  template<typename TYPE>
  inline void show(const TYPE& x){
    cout<<show_if_possible(x,0)<<endl;
  }
  
  /*
  inline string to_string(const vector<int>& v){ // interferes with std::to_string
    ostringstream oss;
    oss<<"(";
    int I=v.size()-1;
    for(int i=0; i<I; i++)
      oss<<v[i]<<",";
    if(v.size()>0) 
      oss<<v[v.size()-1];
    oss<<")";
    return oss.str();
  }
  */ 

  inline ostream& operator<<(ostream& stream, const vector<int>& v){
    stream<<"(";
    int I=v.size()-1;
    for(int i=0; i<I; i++)
      stream<<v[i]<<",";
    if(v.size()>0) 
      stream<<v[v.size()-1];
    stream<<")";
    return stream;
  }

  extern string base_indent;

  struct indenter{
  public:

    string old;

    indenter(const string s){
      old=base_indent;
      base_indent=base_indent+s;
    }

    ~indenter(){
      base_indent=old;
    }

  };

}

#define PRINTL(x) printl(#x,x);
