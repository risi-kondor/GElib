#ifndef _einsum_node
#define _einsum_node

namespace cnine{

  class einsum_node{
  public:

    vector<int> ids;
    vector<shared_ptr<einsum_node> > children;

    int level=0;
    string name="M";

    einsum_node(){}
    
    einsum_node(const int n):
      ids(n,-1){}

    einsum_node(const int n, const string _name):
      ids(n,-1),
      name(_name){}


  public: // ---- Access -------------------------------------------------------------------------------------


    bool contains(const int i){
      //return global_ids.find(i)!=global_ids.end();
      return std::find(ids.begin(),ids.end(),i)!=ids.end();
    }

    int asize(const vector<int>& dims) const{
      int t=0;
      for(auto p:ids)
	t*=dims[p];
      return t;
    }

    virtual int n_ops(const vector<int>& dims) const{
      return 0;
    }



  public: // ---- I/O ----------------------------------------------------------------------------------------


    virtual void latex(ostream& oss) const{
      oss<<name<<"_{";
      for(auto p:ids)
	oss<<static_cast<char>('a'+p);
      oss<<"}";
    }
    
    virtual string index_string() const{
      ostringstream oss;
      for(auto& p:ids)
	oss<<static_cast<char>('a'+p);
      return oss.str();
    }

    virtual string str(const string indent="") const{
      return indent+"["+index_string()+"]"+"\n";
    }

    friend ostream& operator<<(ostream& stream, const einsum_node& x){
      stream<<x.str(); return stream;
    }



  };

}

#endif 
