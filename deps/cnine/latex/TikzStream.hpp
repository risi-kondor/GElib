#ifndef _CnineTikzStream
#define _CnineTikzStream

namespace cnine{

  class TikzStream{
  public:

    int depth=0;
    ostringstream oss;

    void add_line(const string& s){
      oss<<string(2*depth,' ')<<s<<"\n";
    }

    void write(const string& s){
      oss<<string(2*depth,' ')<<s<<"\n";
    }

    TikzStream& operator<<(const string& s){
      oss<<s;
      return *this;
    }

    template<typename TYPE>
    TikzStream& operator<<(const TYPE x){
      oss<<x;
      return *this;
    }
    
    /*
    TikzStream& operator<<(const int x){
      oss<<x;
      return *this;
    }

    TikzStream& operator<<(const float x){
      oss<<x;
      return *this;
    }
    */

    string str(){
      return oss.str();
    }

  };

}

#endif 
