#ifndef _it_instr
#define _it_instr

namespace cnine{

  class it_instr{
  public:

    int target;
    vector<int> sources;


  public: // ---- I/O ---------------------------------------------------------------------------------------


    virtual string indent(const int depth) const{
      return string(2*depth,' ');
    }

    string code(int depth=0) const{
      ostringstream oss;
      oss<<indent(depth)<<"*arr"<<target<<"=";
      for(auto p:sources)
	oss<<"(*arr"<<p<<")*";
      oss<<"\b;\n";
      return oss.str();
    }

  };

}

#endif 
