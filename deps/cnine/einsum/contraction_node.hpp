#ifndef _contraction_node
#define _contraction_node

#include "einsum_node.hpp"


namespace cnine{

  class contraction_node: public einsum_node{
  public:
    
    int id;

    contraction_node(int _id, vector<shared_ptr<einsum_node> >& _children):
      id(_id){
      for(auto& p:_children){
	children.push_back(p);
	for(auto q:p->ids){
	  if(q!=id && !contains(q)){
	    ids.push_back(q);
	  }
	}
	bump(level,p->level);
      }
      level++;
    }


    virtual int n_ops(const vector<int>& dims) const{
      int t=0;
      //for(auto& p: children)
      //t+=p->n_ops(dims); // TODO 
      //return t+dims[id]*adims()*children.size();
      return 0;
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    virtual void latex(ostream& oss) const{
      oss<<"\\Big(\\sum_"<<static_cast<char>('a'+id)<<" ";
      for(auto p: children)
	p->latex(oss);
      oss<<"\\Big)";
    }
    
    virtual string str(const string indent="") const{
      ostringstream oss;
      oss<<indent<<"["<<index_string()<<"] contraction on "<<static_cast<char>('a'+id)<<endl;
      for(auto& p: children)
	oss<<p->str(indent+"  ");
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const contraction_node& x){
      stream<<x.str(); return stream;
    }


  };

}

#endif 
