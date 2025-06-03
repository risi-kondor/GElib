#ifndef _code_env
#define _code_env

#include "Cnine_base.hpp"

namespace cnine{

  class code_env{
  public:

    int depth=0;
    ostringstream oss;

    void add_line(const string& s){
      oss<<string(2*depth,' ')<<s<<"\n";
    }

    void write(const string& s){
      oss<<string(2*depth,' ')<<s<<"\n";
    }

    string str(){
      return oss.str();
    }

  };


  /*
  class for_block{
  public:

    code_env& env;

    for_block(code_env& _env, const int _ix):
      env(_env){
      string ix=to_string(_ix);
      env.add_line("for(int i"+ix+"=0; i"+ix+"<I"+ix+"; i"+ix+"++){");
      env.depth++;
    }

    ~for_block(){
      env.depth--;
      env.add_line("}");
    }

  };


  class for_blocks{
  public:

    code_env& env;
    int n;

    for_blocks(code_env& _env, const vector<int> _ix):
      env(_env), 
      n(_ix.size()){
      for(int i=0; i<n; i++){
	string ix=to_string(_ix[i]);
	env.add_line("for(int i"+ix+"=0; i"+ix+"<I"+ix+"; i"+ix+"++){");
	env.depth++;
      }
    }

    ~for_blocks(){
      for(int i=0; i<n; i++){
	env.depth--;
	env.add_line("}");
      }
    }

  };
  */

}

#endif 
