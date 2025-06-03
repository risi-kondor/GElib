#ifndef _ContractionNode
#define _ContractionNode

#include "EinsumForm.hpp"


namespace cnine{
  namespace einsum{

  class ContractionNode{
  public:

    //shared_ptr<EinsumForm> form;
    
    int contraction_index;
    string contraction_token;
    int arg_id=-1;
    ix_tuple indices;
    ix_set external_indices;
    ix_set internal_indices;

    vector<shared_ptr<ContractionNode> > children;

    int level=0;
    string name="M";

    ContractionNode(){}
    
    ContractionNode(const int _arg_id, const ix_tuple& _indices, const string _name="M"):
      arg_id(_arg_id),
      indices(_indices),
      external_indices(_indices),
      name(_name){}

    ContractionNode(int _contraction_index, const string _token, 
      vector<shared_ptr<ContractionNode> >& _children):
      contraction_index(_contraction_index),
      contraction_token(_token){
      for(auto& p:_children){
	children.push_back(p);
	for(auto q:p->indices){
	  if(q!=contraction_index && !indices.contains(q)){
	    indices.push_back(q);
	  }
	}
	bump(level,p->level);
      }
      level++;
      for(auto p:indices){
	if([&](){for(auto& q:_children)
	       if(!q->indices.contains(p)) return false;
	     return true;}()) external_indices.insert(p); 
	else internal_indices.insert(p);
      }
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    bool contains(const int i) const{
      return indices.contains(i);
    }

    int n_ops(const vector<int>& dims) const{
      int t=0;
      //for(auto& p: children)
      //t+=p->n_ops(dims); // TODO 
      //return t+dims[id]*adims()*children.size();
      return 0;
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string to_string(const int ix) const{
      return string(1,static_cast<char>('a'+ix));
    }

    string index_string() const{
      ostringstream oss;
      for(auto& p:indices)
	oss<<static_cast<char>('a'+p);
      //oss<<form->tokens(p);
      return oss.str();
    }

    void latex(ostream& oss) const{
      oss<<"\\Big(\\sum_"<<to_string(contraction_index)<<" ";
      for(auto p: children)
	p->latex(oss);
      oss<<"\\Big)";
    }
    
    string str(const string indent="") const{
      ostringstream oss;
      if(children.size()>0){
	//oss<<indent<<"contract on "<<to_string(contraction_index)<<": ["<<
	oss<<indent<<"contract on "<<contraction_token<<": ["<<
	  external_indices<<"/"<<internal_indices<<"]"<<endl;
	for(auto& p: children)
	  oss<<p->str(indent+"  ");
      }else{
	oss<<indent<<index_string()<<" ["<<external_indices<<"]"<<endl;
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const ContractionNode& x){
      stream<<x.str(); return stream;
    }


  };

  }
}

#endif 
