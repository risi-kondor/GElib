#ifndef _EtreeParams
#define _EtreeParams

namespace cnine{

  class EtreeParams{
  public:

    vector<int> dimensions;
    //vector<int> strides;

    EtreeParams(const vector<int>& _dimensions):
      dimensions(_dimensions){}


  public: // ---- Access ------------------------------------------------------------------------------------


    int dim(const int i) const{
      return 3;
    }

    Gdims dims(const vector<int> x) const{
      vector<int> R(x.size());
      for(int i=0; i<x.size(); i++)
	R[i]=dimensions[x[i]];
      return R;
    }

    string dims_str(const vector<int> x) const{
      ostringstream oss;
      oss<<"{";
      for(int i=0; i<x.size(); i++)
	oss<<dimensions[x[i]]<<",";
      oss<<"\b}";
      return oss.str();
    }

    int strides(const int T, const int ix) const{
      return 0;
    }

  };

}

#endif 


