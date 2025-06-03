#ifndef _ctree_index_set
#define _ctree_index_set

namespace cnine{

  class ctree_index_set: public set<int>{
  public:
    

    ctree_index_set(){}

    ctree_index_set(const initializer_list<int>& x){
      for(auto& p:x)
	insert(p);
    }

    ctree_index_set(const vector<int>& x){
      for(auto& p:x)
	insert(p);
    }

    operator vector<int>() const{
      vector<int> r;
      for(auto& p:*this)
	r.push_back(p);
      return r;
    }


  public: // ---- Operations --------------------------------------------------------------------------------

    
    void add(const ctree_index_set& x){
      for(auto p:x)
	insert(p);
    }

    ctree_index_set minus(const vector<int>& x) const{
      ctree_index_set r(*this);
      for(auto& p:x)
	if(r.find(p)!=r.end()) r.erase(p);
      return r;
    }

    ctree_index_set contract(const ctree_index_set& y, const int ix) const{
      ctree_index_set r(*this);
      for(auto& p:y) r.insert(p);
      r.erase(ix);
      return r;
    }

    vector<vector<int> > permutations() const{
      vector<vector<int> > r;
      if(size()==1){
	r.push_back(vector<int>({*begin()}));
	return r;
      }
      for(auto& p:*this){
	ctree_index_set sub(*this);
	sub.erase(p);
	auto y=sub.permutations();
	for(auto q:y){
	  q.push_back(p);
	  r.push_back(q);
	}
      }
      return r;
    }


  public: // -----------------------------------------------------------------------------------------------


    string index_str() const{
      ostringstream oss;
      int i=0;
      for(auto& p:*this){
	oss<<"i"<<p;
	if(++i<size()) oss<<",";
      }
      return oss.str();
    }

    string limit_str() const{
      ostringstream oss;
      int i=0;
      for(auto& p:*this){
	oss<<"n"<<p;
	if(++i<size()) oss<<",";
      }
      //if(size()>0) oss<<"\b";
      return oss.str();
    }

  };

}

#endif 
