#ifndef _it_tdef
#define _it_tdef

namespace cnine{

  class it_tdef{
  public:

    int tix;
    index_set indices;

    it_tdef(const int _tix, const index_set& _indices):
      tix(_tix),
      indices(_indices){}


  public: // ---- I/O ---------------------------------------------------------------------------------------


    virtual string indent(const int depth) const{
      return string(2*depth,' ');
    }

    string code(int depth=0) const{
      ostringstream oss;
      oss<<indent(depth)<<"Tensor T"<<tix<<"(";
      for(auto p:indices)
	oss<<"i"<<p<<",";
      oss<<"\b);\n";
      oss<<indent(depth)<<"arr"<<tix<<"=T"<<tix<<".get_arr();\n";
      return oss.str();
    }

  };

}

#endif 
