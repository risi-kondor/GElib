#include "EinsumParams.hpp"

namespace cnine{

  template<typename TYPE>
  void Einsum1_D0_S0_B0(TYPE* r, const TYPE* x, const EsumParams& params){



    TYPE t=0;
    const TYPE* xslice=x+0;
    t+=xslice[0];

    TYPE* rslice=r+0;
    rslice[0]+=t;

  }


  template<typename TYPE>
  void Einsum1_D0_S0_B1(TYPE* r, const TYPE* x, const EsumParams& params){

    int B0=params.sdims[0];
    int rstride_b0=params.rstride_b[0];


    TYPE t=0;
    const TYPE* xslice=x+0;
    t+=xslice[0];

    TYPE* rslice=r+0;
    for(int b0=0; b0<B0; b0++){
      rslice[b0*rstride_b0+0]+=t;
    }

  }


  template<typename TYPE>
  void Einsum1_D0_S0_B2(TYPE* r, const TYPE* x, const EsumParams& params){

    int B0=params.sdims[0];
    int B1=params.sdims[1];
    int rstride_b0=params.rstride_b[0];
    int rstride_b1=params.rstride_b[1];


    TYPE t=0;
    const TYPE* xslice=x+0;
    t+=xslice[0];

    TYPE* rslice=r+0;
    for(int b0=0; b0<B0; b0++){
      for(int b1=0; b1<B1; b1++){
        rslice[b0*rstride_b0+b1*rstride_b1+0]+=t;
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D0_S0_B3(TYPE* r, const TYPE* x, const EsumParams& params){

    int B0=params.sdims[0];
    int B1=params.sdims[1];
    int B2=params.sdims[2];
    int rstride_b0=params.rstride_b[0];
    int rstride_b1=params.rstride_b[1];
    int rstride_b2=params.rstride_b[2];


    TYPE t=0;
    const TYPE* xslice=x+0;
    t+=xslice[0];

    TYPE* rslice=r+0;
    for(int b0=0; b0<B0; b0++){
      for(int b1=0; b1<B1; b1++){
        for(int b2=0; b2<B2; b2++){
          rslice[b0*rstride_b0+b1*rstride_b1+b2*rstride_b2+0]+=t;
        }
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D0_S1_B0(TYPE* r, const TYPE* x, const EsumParams& params){

    int S0=params.sdims[0];
    int xstride_s0=params.xstride_s[0];


    TYPE t=0;
    const TYPE* xslice=x+0;
    for(int s0=0; s0<S0; s0++){
      t+=xslice[s0*xstride_s0+0];
    }

    TYPE* rslice=r+0;
    rslice[0]+=t;

  }


  template<typename TYPE>
  void Einsum1_D0_S1_B1(TYPE* r, const TYPE* x, const EsumParams& params){

    int S0=params.sdims[0];
    int xstride_s0=params.xstride_s[0];
    int B0=params.sdims[0];
    int rstride_b0=params.rstride_b[0];


    TYPE t=0;
    const TYPE* xslice=x+0;
    for(int s0=0; s0<S0; s0++){
      t+=xslice[s0*xstride_s0+0];
    }

    TYPE* rslice=r+0;
    for(int b0=0; b0<B0; b0++){
      rslice[b0*rstride_b0+0]+=t;
    }

  }


  template<typename TYPE>
  void Einsum1_D0_S1_B2(TYPE* r, const TYPE* x, const EsumParams& params){

    int S0=params.sdims[0];
    int xstride_s0=params.xstride_s[0];
    int B0=params.sdims[0];
    int B1=params.sdims[1];
    int rstride_b0=params.rstride_b[0];
    int rstride_b1=params.rstride_b[1];


    TYPE t=0;
    const TYPE* xslice=x+0;
    for(int s0=0; s0<S0; s0++){
      t+=xslice[s0*xstride_s0+0];
    }

    TYPE* rslice=r+0;
    for(int b0=0; b0<B0; b0++){
      for(int b1=0; b1<B1; b1++){
        rslice[b0*rstride_b0+b1*rstride_b1+0]+=t;
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D0_S1_B3(TYPE* r, const TYPE* x, const EsumParams& params){

    int S0=params.sdims[0];
    int xstride_s0=params.xstride_s[0];
    int B0=params.sdims[0];
    int B1=params.sdims[1];
    int B2=params.sdims[2];
    int rstride_b0=params.rstride_b[0];
    int rstride_b1=params.rstride_b[1];
    int rstride_b2=params.rstride_b[2];


    TYPE t=0;
    const TYPE* xslice=x+0;
    for(int s0=0; s0<S0; s0++){
      t+=xslice[s0*xstride_s0+0];
    }

    TYPE* rslice=r+0;
    for(int b0=0; b0<B0; b0++){
      for(int b1=0; b1<B1; b1++){
        for(int b2=0; b2<B2; b2++){
          rslice[b0*rstride_b0+b1*rstride_b1+b2*rstride_b2+0]+=t;
        }
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D0_S2_B0(TYPE* r, const TYPE* x, const EsumParams& params){

    int S0=params.sdims[0];
    int S1=params.sdims[1];
    int xstride_s0=params.xstride_s[0];
    int xstride_s1=params.xstride_s[1];


    TYPE t=0;
    const TYPE* xslice=x+0;
    for(int s0=0; s0<S0; s0++){
      for(int s1=0; s1<S1; s1++){
        t+=xslice[s0*xstride_s0+s1*xstride_s1+0];
      }
    }

    TYPE* rslice=r+0;
    rslice[0]+=t;

  }


  template<typename TYPE>
  void Einsum1_D0_S2_B1(TYPE* r, const TYPE* x, const EsumParams& params){

    int S0=params.sdims[0];
    int S1=params.sdims[1];
    int xstride_s0=params.xstride_s[0];
    int xstride_s1=params.xstride_s[1];
    int B0=params.sdims[0];
    int rstride_b0=params.rstride_b[0];


    TYPE t=0;
    const TYPE* xslice=x+0;
    for(int s0=0; s0<S0; s0++){
      for(int s1=0; s1<S1; s1++){
        t+=xslice[s0*xstride_s0+s1*xstride_s1+0];
      }
    }

    TYPE* rslice=r+0;
    for(int b0=0; b0<B0; b0++){
      rslice[b0*rstride_b0+0]+=t;
    }

  }


  template<typename TYPE>
  void Einsum1_D0_S2_B2(TYPE* r, const TYPE* x, const EsumParams& params){

    int S0=params.sdims[0];
    int S1=params.sdims[1];
    int xstride_s0=params.xstride_s[0];
    int xstride_s1=params.xstride_s[1];
    int B0=params.sdims[0];
    int B1=params.sdims[1];
    int rstride_b0=params.rstride_b[0];
    int rstride_b1=params.rstride_b[1];


    TYPE t=0;
    const TYPE* xslice=x+0;
    for(int s0=0; s0<S0; s0++){
      for(int s1=0; s1<S1; s1++){
        t+=xslice[s0*xstride_s0+s1*xstride_s1+0];
      }
    }

    TYPE* rslice=r+0;
    for(int b0=0; b0<B0; b0++){
      for(int b1=0; b1<B1; b1++){
        rslice[b0*rstride_b0+b1*rstride_b1+0]+=t;
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D0_S2_B3(TYPE* r, const TYPE* x, const EsumParams& params){

    int S0=params.sdims[0];
    int S1=params.sdims[1];
    int xstride_s0=params.xstride_s[0];
    int xstride_s1=params.xstride_s[1];
    int B0=params.sdims[0];
    int B1=params.sdims[1];
    int B2=params.sdims[2];
    int rstride_b0=params.rstride_b[0];
    int rstride_b1=params.rstride_b[1];
    int rstride_b2=params.rstride_b[2];


    TYPE t=0;
    const TYPE* xslice=x+0;
    for(int s0=0; s0<S0; s0++){
      for(int s1=0; s1<S1; s1++){
        t+=xslice[s0*xstride_s0+s1*xstride_s1+0];
      }
    }

    TYPE* rslice=r+0;
    for(int b0=0; b0<B0; b0++){
      for(int b1=0; b1<B1; b1++){
        for(int b2=0; b2<B2; b2++){
          rslice[b0*rstride_b0+b1*rstride_b1+b2*rstride_b2+0]+=t;
        }
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D0_S3_B0(TYPE* r, const TYPE* x, const EsumParams& params){

    int S0=params.sdims[0];
    int S1=params.sdims[1];
    int S2=params.sdims[2];
    int xstride_s0=params.xstride_s[0];
    int xstride_s1=params.xstride_s[1];
    int xstride_s2=params.xstride_s[2];


    TYPE t=0;
    const TYPE* xslice=x+0;
    for(int s0=0; s0<S0; s0++){
      for(int s1=0; s1<S1; s1++){
        for(int s2=0; s2<S2; s2++){
          t+=xslice[s0*xstride_s0+s1*xstride_s1+s2*xstride_s2+0];
        }
      }
    }

    TYPE* rslice=r+0;
    rslice[0]+=t;

  }


  template<typename TYPE>
  void Einsum1_D0_S3_B1(TYPE* r, const TYPE* x, const EsumParams& params){

    int S0=params.sdims[0];
    int S1=params.sdims[1];
    int S2=params.sdims[2];
    int xstride_s0=params.xstride_s[0];
    int xstride_s1=params.xstride_s[1];
    int xstride_s2=params.xstride_s[2];
    int B0=params.sdims[0];
    int rstride_b0=params.rstride_b[0];


    TYPE t=0;
    const TYPE* xslice=x+0;
    for(int s0=0; s0<S0; s0++){
      for(int s1=0; s1<S1; s1++){
        for(int s2=0; s2<S2; s2++){
          t+=xslice[s0*xstride_s0+s1*xstride_s1+s2*xstride_s2+0];
        }
      }
    }

    TYPE* rslice=r+0;
    for(int b0=0; b0<B0; b0++){
      rslice[b0*rstride_b0+0]+=t;
    }

  }


  template<typename TYPE>
  void Einsum1_D0_S3_B2(TYPE* r, const TYPE* x, const EsumParams& params){

    int S0=params.sdims[0];
    int S1=params.sdims[1];
    int S2=params.sdims[2];
    int xstride_s0=params.xstride_s[0];
    int xstride_s1=params.xstride_s[1];
    int xstride_s2=params.xstride_s[2];
    int B0=params.sdims[0];
    int B1=params.sdims[1];
    int rstride_b0=params.rstride_b[0];
    int rstride_b1=params.rstride_b[1];


    TYPE t=0;
    const TYPE* xslice=x+0;
    for(int s0=0; s0<S0; s0++){
      for(int s1=0; s1<S1; s1++){
        for(int s2=0; s2<S2; s2++){
          t+=xslice[s0*xstride_s0+s1*xstride_s1+s2*xstride_s2+0];
        }
      }
    }

    TYPE* rslice=r+0;
    for(int b0=0; b0<B0; b0++){
      for(int b1=0; b1<B1; b1++){
        rslice[b0*rstride_b0+b1*rstride_b1+0]+=t;
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D0_S3_B3(TYPE* r, const TYPE* x, const EsumParams& params){

    int S0=params.sdims[0];
    int S1=params.sdims[1];
    int S2=params.sdims[2];
    int xstride_s0=params.xstride_s[0];
    int xstride_s1=params.xstride_s[1];
    int xstride_s2=params.xstride_s[2];
    int B0=params.sdims[0];
    int B1=params.sdims[1];
    int B2=params.sdims[2];
    int rstride_b0=params.rstride_b[0];
    int rstride_b1=params.rstride_b[1];
    int rstride_b2=params.rstride_b[2];


    TYPE t=0;
    const TYPE* xslice=x+0;
    for(int s0=0; s0<S0; s0++){
      for(int s1=0; s1<S1; s1++){
        for(int s2=0; s2<S2; s2++){
          t+=xslice[s0*xstride_s0+s1*xstride_s1+s2*xstride_s2+0];
        }
      }
    }

    TYPE* rslice=r+0;
    for(int b0=0; b0<B0; b0++){
      for(int b1=0; b1<B1; b1++){
        for(int b2=0; b2<B2; b2++){
          rslice[b0*rstride_b0+b1*rstride_b1+b2*rstride_b2+0]+=t;
        }
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D1_S0_B0(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int xstride_d0=params.xstride_d[0];
    int rstride_d0=params.rstride_d[0];

    for(int i0=0; i0<I0; i0++){

      TYPE t=0;
      const TYPE* xslice=x+i0*xstride_d0+0;
      t+=xslice[0];

      TYPE* rslice=r+i0*rstride_d0+0;
      rslice[0]+=t;
    }

  }


  template<typename TYPE>
  void Einsum1_D1_S0_B1(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int xstride_d0=params.xstride_d[0];
    int rstride_d0=params.rstride_d[0];
    int B0=params.sdims[0];
    int rstride_b0=params.rstride_b[0];

    for(int i0=0; i0<I0; i0++){

      TYPE t=0;
      const TYPE* xslice=x+i0*xstride_d0+0;
      t+=xslice[0];

      TYPE* rslice=r+i0*rstride_d0+0;
      for(int b0=0; b0<B0; b0++){
        rslice[b0*rstride_b0+0]+=t;
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D1_S0_B2(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int xstride_d0=params.xstride_d[0];
    int rstride_d0=params.rstride_d[0];
    int B0=params.sdims[0];
    int B1=params.sdims[1];
    int rstride_b0=params.rstride_b[0];
    int rstride_b1=params.rstride_b[1];

    for(int i0=0; i0<I0; i0++){

      TYPE t=0;
      const TYPE* xslice=x+i0*xstride_d0+0;
      t+=xslice[0];

      TYPE* rslice=r+i0*rstride_d0+0;
      for(int b0=0; b0<B0; b0++){
        for(int b1=0; b1<B1; b1++){
          rslice[b0*rstride_b0+b1*rstride_b1+0]+=t;
        }
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D1_S0_B3(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int xstride_d0=params.xstride_d[0];
    int rstride_d0=params.rstride_d[0];
    int B0=params.sdims[0];
    int B1=params.sdims[1];
    int B2=params.sdims[2];
    int rstride_b0=params.rstride_b[0];
    int rstride_b1=params.rstride_b[1];
    int rstride_b2=params.rstride_b[2];

    for(int i0=0; i0<I0; i0++){

      TYPE t=0;
      const TYPE* xslice=x+i0*xstride_d0+0;
      t+=xslice[0];

      TYPE* rslice=r+i0*rstride_d0+0;
      for(int b0=0; b0<B0; b0++){
        for(int b1=0; b1<B1; b1++){
          for(int b2=0; b2<B2; b2++){
            rslice[b0*rstride_b0+b1*rstride_b1+b2*rstride_b2+0]+=t;
          }
        }
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D1_S1_B0(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int xstride_d0=params.xstride_d[0];
    int rstride_d0=params.rstride_d[0];
    int S0=params.sdims[0];
    int xstride_s0=params.xstride_s[0];

    for(int i0=0; i0<I0; i0++){

      TYPE t=0;
      const TYPE* xslice=x+i0*xstride_d0+0;
      for(int s0=0; s0<S0; s0++){
        t+=xslice[s0*xstride_s0+0];
      }

      TYPE* rslice=r+i0*rstride_d0+0;
      rslice[0]+=t;
    }

  }


  template<typename TYPE>
  void Einsum1_D1_S1_B1(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int xstride_d0=params.xstride_d[0];
    int rstride_d0=params.rstride_d[0];
    int S0=params.sdims[0];
    int xstride_s0=params.xstride_s[0];
    int B0=params.sdims[0];
    int rstride_b0=params.rstride_b[0];

    for(int i0=0; i0<I0; i0++){

      TYPE t=0;
      const TYPE* xslice=x+i0*xstride_d0+0;
      for(int s0=0; s0<S0; s0++){
        t+=xslice[s0*xstride_s0+0];
      }

      TYPE* rslice=r+i0*rstride_d0+0;
      for(int b0=0; b0<B0; b0++){
        rslice[b0*rstride_b0+0]+=t;
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D1_S1_B2(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int xstride_d0=params.xstride_d[0];
    int rstride_d0=params.rstride_d[0];
    int S0=params.sdims[0];
    int xstride_s0=params.xstride_s[0];
    int B0=params.sdims[0];
    int B1=params.sdims[1];
    int rstride_b0=params.rstride_b[0];
    int rstride_b1=params.rstride_b[1];

    for(int i0=0; i0<I0; i0++){

      TYPE t=0;
      const TYPE* xslice=x+i0*xstride_d0+0;
      for(int s0=0; s0<S0; s0++){
        t+=xslice[s0*xstride_s0+0];
      }

      TYPE* rslice=r+i0*rstride_d0+0;
      for(int b0=0; b0<B0; b0++){
        for(int b1=0; b1<B1; b1++){
          rslice[b0*rstride_b0+b1*rstride_b1+0]+=t;
        }
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D1_S1_B3(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int xstride_d0=params.xstride_d[0];
    int rstride_d0=params.rstride_d[0];
    int S0=params.sdims[0];
    int xstride_s0=params.xstride_s[0];
    int B0=params.sdims[0];
    int B1=params.sdims[1];
    int B2=params.sdims[2];
    int rstride_b0=params.rstride_b[0];
    int rstride_b1=params.rstride_b[1];
    int rstride_b2=params.rstride_b[2];

    for(int i0=0; i0<I0; i0++){

      TYPE t=0;
      const TYPE* xslice=x+i0*xstride_d0+0;
      for(int s0=0; s0<S0; s0++){
        t+=xslice[s0*xstride_s0+0];
      }

      TYPE* rslice=r+i0*rstride_d0+0;
      for(int b0=0; b0<B0; b0++){
        for(int b1=0; b1<B1; b1++){
          for(int b2=0; b2<B2; b2++){
            rslice[b0*rstride_b0+b1*rstride_b1+b2*rstride_b2+0]+=t;
          }
        }
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D1_S2_B0(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int xstride_d0=params.xstride_d[0];
    int rstride_d0=params.rstride_d[0];
    int S0=params.sdims[0];
    int S1=params.sdims[1];
    int xstride_s0=params.xstride_s[0];
    int xstride_s1=params.xstride_s[1];

    for(int i0=0; i0<I0; i0++){

      TYPE t=0;
      const TYPE* xslice=x+i0*xstride_d0+0;
      for(int s0=0; s0<S0; s0++){
        for(int s1=0; s1<S1; s1++){
          t+=xslice[s0*xstride_s0+s1*xstride_s1+0];
        }
      }

      TYPE* rslice=r+i0*rstride_d0+0;
      rslice[0]+=t;
    }

  }


  template<typename TYPE>
  void Einsum1_D1_S2_B1(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int xstride_d0=params.xstride_d[0];
    int rstride_d0=params.rstride_d[0];
    int S0=params.sdims[0];
    int S1=params.sdims[1];
    int xstride_s0=params.xstride_s[0];
    int xstride_s1=params.xstride_s[1];
    int B0=params.sdims[0];
    int rstride_b0=params.rstride_b[0];

    for(int i0=0; i0<I0; i0++){

      TYPE t=0;
      const TYPE* xslice=x+i0*xstride_d0+0;
      for(int s0=0; s0<S0; s0++){
        for(int s1=0; s1<S1; s1++){
          t+=xslice[s0*xstride_s0+s1*xstride_s1+0];
        }
      }

      TYPE* rslice=r+i0*rstride_d0+0;
      for(int b0=0; b0<B0; b0++){
        rslice[b0*rstride_b0+0]+=t;
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D1_S2_B2(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int xstride_d0=params.xstride_d[0];
    int rstride_d0=params.rstride_d[0];
    int S0=params.sdims[0];
    int S1=params.sdims[1];
    int xstride_s0=params.xstride_s[0];
    int xstride_s1=params.xstride_s[1];
    int B0=params.sdims[0];
    int B1=params.sdims[1];
    int rstride_b0=params.rstride_b[0];
    int rstride_b1=params.rstride_b[1];

    for(int i0=0; i0<I0; i0++){

      TYPE t=0;
      const TYPE* xslice=x+i0*xstride_d0+0;
      for(int s0=0; s0<S0; s0++){
        for(int s1=0; s1<S1; s1++){
          t+=xslice[s0*xstride_s0+s1*xstride_s1+0];
        }
      }

      TYPE* rslice=r+i0*rstride_d0+0;
      for(int b0=0; b0<B0; b0++){
        for(int b1=0; b1<B1; b1++){
          rslice[b0*rstride_b0+b1*rstride_b1+0]+=t;
        }
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D1_S2_B3(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int xstride_d0=params.xstride_d[0];
    int rstride_d0=params.rstride_d[0];
    int S0=params.sdims[0];
    int S1=params.sdims[1];
    int xstride_s0=params.xstride_s[0];
    int xstride_s1=params.xstride_s[1];
    int B0=params.sdims[0];
    int B1=params.sdims[1];
    int B2=params.sdims[2];
    int rstride_b0=params.rstride_b[0];
    int rstride_b1=params.rstride_b[1];
    int rstride_b2=params.rstride_b[2];

    for(int i0=0; i0<I0; i0++){

      TYPE t=0;
      const TYPE* xslice=x+i0*xstride_d0+0;
      for(int s0=0; s0<S0; s0++){
        for(int s1=0; s1<S1; s1++){
          t+=xslice[s0*xstride_s0+s1*xstride_s1+0];
        }
      }

      TYPE* rslice=r+i0*rstride_d0+0;
      for(int b0=0; b0<B0; b0++){
        for(int b1=0; b1<B1; b1++){
          for(int b2=0; b2<B2; b2++){
            rslice[b0*rstride_b0+b1*rstride_b1+b2*rstride_b2+0]+=t;
          }
        }
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D1_S3_B0(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int xstride_d0=params.xstride_d[0];
    int rstride_d0=params.rstride_d[0];
    int S0=params.sdims[0];
    int S1=params.sdims[1];
    int S2=params.sdims[2];
    int xstride_s0=params.xstride_s[0];
    int xstride_s1=params.xstride_s[1];
    int xstride_s2=params.xstride_s[2];

    for(int i0=0; i0<I0; i0++){

      TYPE t=0;
      const TYPE* xslice=x+i0*xstride_d0+0;
      for(int s0=0; s0<S0; s0++){
        for(int s1=0; s1<S1; s1++){
          for(int s2=0; s2<S2; s2++){
            t+=xslice[s0*xstride_s0+s1*xstride_s1+s2*xstride_s2+0];
          }
        }
      }

      TYPE* rslice=r+i0*rstride_d0+0;
      rslice[0]+=t;
    }

  }


  template<typename TYPE>
  void Einsum1_D1_S3_B1(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int xstride_d0=params.xstride_d[0];
    int rstride_d0=params.rstride_d[0];
    int S0=params.sdims[0];
    int S1=params.sdims[1];
    int S2=params.sdims[2];
    int xstride_s0=params.xstride_s[0];
    int xstride_s1=params.xstride_s[1];
    int xstride_s2=params.xstride_s[2];
    int B0=params.sdims[0];
    int rstride_b0=params.rstride_b[0];

    for(int i0=0; i0<I0; i0++){

      TYPE t=0;
      const TYPE* xslice=x+i0*xstride_d0+0;
      for(int s0=0; s0<S0; s0++){
        for(int s1=0; s1<S1; s1++){
          for(int s2=0; s2<S2; s2++){
            t+=xslice[s0*xstride_s0+s1*xstride_s1+s2*xstride_s2+0];
          }
        }
      }

      TYPE* rslice=r+i0*rstride_d0+0;
      for(int b0=0; b0<B0; b0++){
        rslice[b0*rstride_b0+0]+=t;
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D1_S3_B2(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int xstride_d0=params.xstride_d[0];
    int rstride_d0=params.rstride_d[0];
    int S0=params.sdims[0];
    int S1=params.sdims[1];
    int S2=params.sdims[2];
    int xstride_s0=params.xstride_s[0];
    int xstride_s1=params.xstride_s[1];
    int xstride_s2=params.xstride_s[2];
    int B0=params.sdims[0];
    int B1=params.sdims[1];
    int rstride_b0=params.rstride_b[0];
    int rstride_b1=params.rstride_b[1];

    for(int i0=0; i0<I0; i0++){

      TYPE t=0;
      const TYPE* xslice=x+i0*xstride_d0+0;
      for(int s0=0; s0<S0; s0++){
        for(int s1=0; s1<S1; s1++){
          for(int s2=0; s2<S2; s2++){
            t+=xslice[s0*xstride_s0+s1*xstride_s1+s2*xstride_s2+0];
          }
        }
      }

      TYPE* rslice=r+i0*rstride_d0+0;
      for(int b0=0; b0<B0; b0++){
        for(int b1=0; b1<B1; b1++){
          rslice[b0*rstride_b0+b1*rstride_b1+0]+=t;
        }
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D1_S3_B3(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int xstride_d0=params.xstride_d[0];
    int rstride_d0=params.rstride_d[0];
    int S0=params.sdims[0];
    int S1=params.sdims[1];
    int S2=params.sdims[2];
    int xstride_s0=params.xstride_s[0];
    int xstride_s1=params.xstride_s[1];
    int xstride_s2=params.xstride_s[2];
    int B0=params.sdims[0];
    int B1=params.sdims[1];
    int B2=params.sdims[2];
    int rstride_b0=params.rstride_b[0];
    int rstride_b1=params.rstride_b[1];
    int rstride_b2=params.rstride_b[2];

    for(int i0=0; i0<I0; i0++){

      TYPE t=0;
      const TYPE* xslice=x+i0*xstride_d0+0;
      for(int s0=0; s0<S0; s0++){
        for(int s1=0; s1<S1; s1++){
          for(int s2=0; s2<S2; s2++){
            t+=xslice[s0*xstride_s0+s1*xstride_s1+s2*xstride_s2+0];
          }
        }
      }

      TYPE* rslice=r+i0*rstride_d0+0;
      for(int b0=0; b0<B0; b0++){
        for(int b1=0; b1<B1; b1++){
          for(int b2=0; b2<B2; b2++){
            rslice[b0*rstride_b0+b1*rstride_b1+b2*rstride_b2+0]+=t;
          }
        }
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D2_S0_B0(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int I1=params.ddims[1];
    int xstride_d0=params.xstride_d[0];
    int xstride_d1=params.xstride_d[1];
    int rstride_d0=params.rstride_d[0];
    int rstride_d1=params.rstride_d[1];

    for(int i0=0; i0<I0; i0++){
      for(int i1=0; i1<I1; i1++){

        TYPE t=0;
        const TYPE* xslice=x+i0*xstride_d0+i1*xstride_d1+0;
        t+=xslice[0];

        TYPE* rslice=r+i0*rstride_d0+i1*rstride_d1+0;
        rslice[0]+=t;
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D2_S0_B1(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int I1=params.ddims[1];
    int xstride_d0=params.xstride_d[0];
    int xstride_d1=params.xstride_d[1];
    int rstride_d0=params.rstride_d[0];
    int rstride_d1=params.rstride_d[1];
    int B0=params.sdims[0];
    int rstride_b0=params.rstride_b[0];

    for(int i0=0; i0<I0; i0++){
      for(int i1=0; i1<I1; i1++){

        TYPE t=0;
        const TYPE* xslice=x+i0*xstride_d0+i1*xstride_d1+0;
        t+=xslice[0];

        TYPE* rslice=r+i0*rstride_d0+i1*rstride_d1+0;
        for(int b0=0; b0<B0; b0++){
          rslice[b0*rstride_b0+0]+=t;
        }
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D2_S0_B2(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int I1=params.ddims[1];
    int xstride_d0=params.xstride_d[0];
    int xstride_d1=params.xstride_d[1];
    int rstride_d0=params.rstride_d[0];
    int rstride_d1=params.rstride_d[1];
    int B0=params.sdims[0];
    int B1=params.sdims[1];
    int rstride_b0=params.rstride_b[0];
    int rstride_b1=params.rstride_b[1];

    for(int i0=0; i0<I0; i0++){
      for(int i1=0; i1<I1; i1++){

        TYPE t=0;
        const TYPE* xslice=x+i0*xstride_d0+i1*xstride_d1+0;
        t+=xslice[0];

        TYPE* rslice=r+i0*rstride_d0+i1*rstride_d1+0;
        for(int b0=0; b0<B0; b0++){
          for(int b1=0; b1<B1; b1++){
            rslice[b0*rstride_b0+b1*rstride_b1+0]+=t;
          }
        }
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D2_S0_B3(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int I1=params.ddims[1];
    int xstride_d0=params.xstride_d[0];
    int xstride_d1=params.xstride_d[1];
    int rstride_d0=params.rstride_d[0];
    int rstride_d1=params.rstride_d[1];
    int B0=params.sdims[0];
    int B1=params.sdims[1];
    int B2=params.sdims[2];
    int rstride_b0=params.rstride_b[0];
    int rstride_b1=params.rstride_b[1];
    int rstride_b2=params.rstride_b[2];

    for(int i0=0; i0<I0; i0++){
      for(int i1=0; i1<I1; i1++){

        TYPE t=0;
        const TYPE* xslice=x+i0*xstride_d0+i1*xstride_d1+0;
        t+=xslice[0];

        TYPE* rslice=r+i0*rstride_d0+i1*rstride_d1+0;
        for(int b0=0; b0<B0; b0++){
          for(int b1=0; b1<B1; b1++){
            for(int b2=0; b2<B2; b2++){
              rslice[b0*rstride_b0+b1*rstride_b1+b2*rstride_b2+0]+=t;
            }
          }
        }
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D2_S1_B0(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int I1=params.ddims[1];
    int xstride_d0=params.xstride_d[0];
    int xstride_d1=params.xstride_d[1];
    int rstride_d0=params.rstride_d[0];
    int rstride_d1=params.rstride_d[1];
    int S0=params.sdims[0];
    int xstride_s0=params.xstride_s[0];

    for(int i0=0; i0<I0; i0++){
      for(int i1=0; i1<I1; i1++){

        TYPE t=0;
        const TYPE* xslice=x+i0*xstride_d0+i1*xstride_d1+0;
        for(int s0=0; s0<S0; s0++){
          t+=xslice[s0*xstride_s0+0];
        }

        TYPE* rslice=r+i0*rstride_d0+i1*rstride_d1+0;
        rslice[0]+=t;
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D2_S1_B1(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int I1=params.ddims[1];
    int xstride_d0=params.xstride_d[0];
    int xstride_d1=params.xstride_d[1];
    int rstride_d0=params.rstride_d[0];
    int rstride_d1=params.rstride_d[1];
    int S0=params.sdims[0];
    int xstride_s0=params.xstride_s[0];
    int B0=params.sdims[0];
    int rstride_b0=params.rstride_b[0];

    for(int i0=0; i0<I0; i0++){
      for(int i1=0; i1<I1; i1++){

        TYPE t=0;
        const TYPE* xslice=x+i0*xstride_d0+i1*xstride_d1+0;
        for(int s0=0; s0<S0; s0++){
          t+=xslice[s0*xstride_s0+0];
        }

        TYPE* rslice=r+i0*rstride_d0+i1*rstride_d1+0;
        for(int b0=0; b0<B0; b0++){
          rslice[b0*rstride_b0+0]+=t;
        }
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D2_S1_B2(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int I1=params.ddims[1];
    int xstride_d0=params.xstride_d[0];
    int xstride_d1=params.xstride_d[1];
    int rstride_d0=params.rstride_d[0];
    int rstride_d1=params.rstride_d[1];
    int S0=params.sdims[0];
    int xstride_s0=params.xstride_s[0];
    int B0=params.sdims[0];
    int B1=params.sdims[1];
    int rstride_b0=params.rstride_b[0];
    int rstride_b1=params.rstride_b[1];

    for(int i0=0; i0<I0; i0++){
      for(int i1=0; i1<I1; i1++){

        TYPE t=0;
        const TYPE* xslice=x+i0*xstride_d0+i1*xstride_d1+0;
        for(int s0=0; s0<S0; s0++){
          t+=xslice[s0*xstride_s0+0];
        }

        TYPE* rslice=r+i0*rstride_d0+i1*rstride_d1+0;
        for(int b0=0; b0<B0; b0++){
          for(int b1=0; b1<B1; b1++){
            rslice[b0*rstride_b0+b1*rstride_b1+0]+=t;
          }
        }
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D2_S1_B3(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int I1=params.ddims[1];
    int xstride_d0=params.xstride_d[0];
    int xstride_d1=params.xstride_d[1];
    int rstride_d0=params.rstride_d[0];
    int rstride_d1=params.rstride_d[1];
    int S0=params.sdims[0];
    int xstride_s0=params.xstride_s[0];
    int B0=params.sdims[0];
    int B1=params.sdims[1];
    int B2=params.sdims[2];
    int rstride_b0=params.rstride_b[0];
    int rstride_b1=params.rstride_b[1];
    int rstride_b2=params.rstride_b[2];

    for(int i0=0; i0<I0; i0++){
      for(int i1=0; i1<I1; i1++){

        TYPE t=0;
        const TYPE* xslice=x+i0*xstride_d0+i1*xstride_d1+0;
        for(int s0=0; s0<S0; s0++){
          t+=xslice[s0*xstride_s0+0];
        }

        TYPE* rslice=r+i0*rstride_d0+i1*rstride_d1+0;
        for(int b0=0; b0<B0; b0++){
          for(int b1=0; b1<B1; b1++){
            for(int b2=0; b2<B2; b2++){
              rslice[b0*rstride_b0+b1*rstride_b1+b2*rstride_b2+0]+=t;
            }
          }
        }
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D2_S2_B0(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int I1=params.ddims[1];
    int xstride_d0=params.xstride_d[0];
    int xstride_d1=params.xstride_d[1];
    int rstride_d0=params.rstride_d[0];
    int rstride_d1=params.rstride_d[1];
    int S0=params.sdims[0];
    int S1=params.sdims[1];
    int xstride_s0=params.xstride_s[0];
    int xstride_s1=params.xstride_s[1];

    for(int i0=0; i0<I0; i0++){
      for(int i1=0; i1<I1; i1++){

        TYPE t=0;
        const TYPE* xslice=x+i0*xstride_d0+i1*xstride_d1+0;
        for(int s0=0; s0<S0; s0++){
          for(int s1=0; s1<S1; s1++){
            t+=xslice[s0*xstride_s0+s1*xstride_s1+0];
          }
        }

        TYPE* rslice=r+i0*rstride_d0+i1*rstride_d1+0;
        rslice[0]+=t;
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D2_S2_B1(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int I1=params.ddims[1];
    int xstride_d0=params.xstride_d[0];
    int xstride_d1=params.xstride_d[1];
    int rstride_d0=params.rstride_d[0];
    int rstride_d1=params.rstride_d[1];
    int S0=params.sdims[0];
    int S1=params.sdims[1];
    int xstride_s0=params.xstride_s[0];
    int xstride_s1=params.xstride_s[1];
    int B0=params.sdims[0];
    int rstride_b0=params.rstride_b[0];

    for(int i0=0; i0<I0; i0++){
      for(int i1=0; i1<I1; i1++){

        TYPE t=0;
        const TYPE* xslice=x+i0*xstride_d0+i1*xstride_d1+0;
        for(int s0=0; s0<S0; s0++){
          for(int s1=0; s1<S1; s1++){
            t+=xslice[s0*xstride_s0+s1*xstride_s1+0];
          }
        }

        TYPE* rslice=r+i0*rstride_d0+i1*rstride_d1+0;
        for(int b0=0; b0<B0; b0++){
          rslice[b0*rstride_b0+0]+=t;
        }
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D2_S2_B2(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int I1=params.ddims[1];
    int xstride_d0=params.xstride_d[0];
    int xstride_d1=params.xstride_d[1];
    int rstride_d0=params.rstride_d[0];
    int rstride_d1=params.rstride_d[1];
    int S0=params.sdims[0];
    int S1=params.sdims[1];
    int xstride_s0=params.xstride_s[0];
    int xstride_s1=params.xstride_s[1];
    int B0=params.sdims[0];
    int B1=params.sdims[1];
    int rstride_b0=params.rstride_b[0];
    int rstride_b1=params.rstride_b[1];

    for(int i0=0; i0<I0; i0++){
      for(int i1=0; i1<I1; i1++){

        TYPE t=0;
        const TYPE* xslice=x+i0*xstride_d0+i1*xstride_d1+0;
        for(int s0=0; s0<S0; s0++){
          for(int s1=0; s1<S1; s1++){
            t+=xslice[s0*xstride_s0+s1*xstride_s1+0];
          }
        }

        TYPE* rslice=r+i0*rstride_d0+i1*rstride_d1+0;
        for(int b0=0; b0<B0; b0++){
          for(int b1=0; b1<B1; b1++){
            rslice[b0*rstride_b0+b1*rstride_b1+0]+=t;
          }
        }
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D2_S2_B3(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int I1=params.ddims[1];
    int xstride_d0=params.xstride_d[0];
    int xstride_d1=params.xstride_d[1];
    int rstride_d0=params.rstride_d[0];
    int rstride_d1=params.rstride_d[1];
    int S0=params.sdims[0];
    int S1=params.sdims[1];
    int xstride_s0=params.xstride_s[0];
    int xstride_s1=params.xstride_s[1];
    int B0=params.sdims[0];
    int B1=params.sdims[1];
    int B2=params.sdims[2];
    int rstride_b0=params.rstride_b[0];
    int rstride_b1=params.rstride_b[1];
    int rstride_b2=params.rstride_b[2];

    for(int i0=0; i0<I0; i0++){
      for(int i1=0; i1<I1; i1++){

        TYPE t=0;
        const TYPE* xslice=x+i0*xstride_d0+i1*xstride_d1+0;
        for(int s0=0; s0<S0; s0++){
          for(int s1=0; s1<S1; s1++){
            t+=xslice[s0*xstride_s0+s1*xstride_s1+0];
          }
        }

        TYPE* rslice=r+i0*rstride_d0+i1*rstride_d1+0;
        for(int b0=0; b0<B0; b0++){
          for(int b1=0; b1<B1; b1++){
            for(int b2=0; b2<B2; b2++){
              rslice[b0*rstride_b0+b1*rstride_b1+b2*rstride_b2+0]+=t;
            }
          }
        }
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D2_S3_B0(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int I1=params.ddims[1];
    int xstride_d0=params.xstride_d[0];
    int xstride_d1=params.xstride_d[1];
    int rstride_d0=params.rstride_d[0];
    int rstride_d1=params.rstride_d[1];
    int S0=params.sdims[0];
    int S1=params.sdims[1];
    int S2=params.sdims[2];
    int xstride_s0=params.xstride_s[0];
    int xstride_s1=params.xstride_s[1];
    int xstride_s2=params.xstride_s[2];

    for(int i0=0; i0<I0; i0++){
      for(int i1=0; i1<I1; i1++){

        TYPE t=0;
        const TYPE* xslice=x+i0*xstride_d0+i1*xstride_d1+0;
        for(int s0=0; s0<S0; s0++){
          for(int s1=0; s1<S1; s1++){
            for(int s2=0; s2<S2; s2++){
              t+=xslice[s0*xstride_s0+s1*xstride_s1+s2*xstride_s2+0];
            }
          }
        }

        TYPE* rslice=r+i0*rstride_d0+i1*rstride_d1+0;
        rslice[0]+=t;
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D2_S3_B1(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int I1=params.ddims[1];
    int xstride_d0=params.xstride_d[0];
    int xstride_d1=params.xstride_d[1];
    int rstride_d0=params.rstride_d[0];
    int rstride_d1=params.rstride_d[1];
    int S0=params.sdims[0];
    int S1=params.sdims[1];
    int S2=params.sdims[2];
    int xstride_s0=params.xstride_s[0];
    int xstride_s1=params.xstride_s[1];
    int xstride_s2=params.xstride_s[2];
    int B0=params.sdims[0];
    int rstride_b0=params.rstride_b[0];

    for(int i0=0; i0<I0; i0++){
      for(int i1=0; i1<I1; i1++){

        TYPE t=0;
        const TYPE* xslice=x+i0*xstride_d0+i1*xstride_d1+0;
        for(int s0=0; s0<S0; s0++){
          for(int s1=0; s1<S1; s1++){
            for(int s2=0; s2<S2; s2++){
              t+=xslice[s0*xstride_s0+s1*xstride_s1+s2*xstride_s2+0];
            }
          }
        }

        TYPE* rslice=r+i0*rstride_d0+i1*rstride_d1+0;
        for(int b0=0; b0<B0; b0++){
          rslice[b0*rstride_b0+0]+=t;
        }
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D2_S3_B2(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int I1=params.ddims[1];
    int xstride_d0=params.xstride_d[0];
    int xstride_d1=params.xstride_d[1];
    int rstride_d0=params.rstride_d[0];
    int rstride_d1=params.rstride_d[1];
    int S0=params.sdims[0];
    int S1=params.sdims[1];
    int S2=params.sdims[2];
    int xstride_s0=params.xstride_s[0];
    int xstride_s1=params.xstride_s[1];
    int xstride_s2=params.xstride_s[2];
    int B0=params.sdims[0];
    int B1=params.sdims[1];
    int rstride_b0=params.rstride_b[0];
    int rstride_b1=params.rstride_b[1];

    for(int i0=0; i0<I0; i0++){
      for(int i1=0; i1<I1; i1++){

        TYPE t=0;
        const TYPE* xslice=x+i0*xstride_d0+i1*xstride_d1+0;
        for(int s0=0; s0<S0; s0++){
          for(int s1=0; s1<S1; s1++){
            for(int s2=0; s2<S2; s2++){
              t+=xslice[s0*xstride_s0+s1*xstride_s1+s2*xstride_s2+0];
            }
          }
        }

        TYPE* rslice=r+i0*rstride_d0+i1*rstride_d1+0;
        for(int b0=0; b0<B0; b0++){
          for(int b1=0; b1<B1; b1++){
            rslice[b0*rstride_b0+b1*rstride_b1+0]+=t;
          }
        }
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D2_S3_B3(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int I1=params.ddims[1];
    int xstride_d0=params.xstride_d[0];
    int xstride_d1=params.xstride_d[1];
    int rstride_d0=params.rstride_d[0];
    int rstride_d1=params.rstride_d[1];
    int S0=params.sdims[0];
    int S1=params.sdims[1];
    int S2=params.sdims[2];
    int xstride_s0=params.xstride_s[0];
    int xstride_s1=params.xstride_s[1];
    int xstride_s2=params.xstride_s[2];
    int B0=params.sdims[0];
    int B1=params.sdims[1];
    int B2=params.sdims[2];
    int rstride_b0=params.rstride_b[0];
    int rstride_b1=params.rstride_b[1];
    int rstride_b2=params.rstride_b[2];

    for(int i0=0; i0<I0; i0++){
      for(int i1=0; i1<I1; i1++){

        TYPE t=0;
        const TYPE* xslice=x+i0*xstride_d0+i1*xstride_d1+0;
        for(int s0=0; s0<S0; s0++){
          for(int s1=0; s1<S1; s1++){
            for(int s2=0; s2<S2; s2++){
              t+=xslice[s0*xstride_s0+s1*xstride_s1+s2*xstride_s2+0];
            }
          }
        }

        TYPE* rslice=r+i0*rstride_d0+i1*rstride_d1+0;
        for(int b0=0; b0<B0; b0++){
          for(int b1=0; b1<B1; b1++){
            for(int b2=0; b2<B2; b2++){
              rslice[b0*rstride_b0+b1*rstride_b1+b2*rstride_b2+0]+=t;
            }
          }
        }
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D3_S0_B0(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int I1=params.ddims[1];
    int I2=params.ddims[2];
    int xstride_d0=params.xstride_d[0];
    int xstride_d1=params.xstride_d[1];
    int xstride_d2=params.xstride_d[2];
    int rstride_d0=params.rstride_d[0];
    int rstride_d1=params.rstride_d[1];
    int rstride_d2=params.rstride_d[2];

    for(int i0=0; i0<I0; i0++){
      for(int i1=0; i1<I1; i1++){
        for(int i2=0; i2<I2; i2++){

          TYPE t=0;
          const TYPE* xslice=x+i0*xstride_d0+i1*xstride_d1+i2*xstride_d2+0;
          t+=xslice[0];

          TYPE* rslice=r+i0*rstride_d0+i1*rstride_d1+i2*rstride_d2+0;
          rslice[0]+=t;
        }
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D3_S0_B1(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int I1=params.ddims[1];
    int I2=params.ddims[2];
    int xstride_d0=params.xstride_d[0];
    int xstride_d1=params.xstride_d[1];
    int xstride_d2=params.xstride_d[2];
    int rstride_d0=params.rstride_d[0];
    int rstride_d1=params.rstride_d[1];
    int rstride_d2=params.rstride_d[2];
    int B0=params.sdims[0];
    int rstride_b0=params.rstride_b[0];

    for(int i0=0; i0<I0; i0++){
      for(int i1=0; i1<I1; i1++){
        for(int i2=0; i2<I2; i2++){

          TYPE t=0;
          const TYPE* xslice=x+i0*xstride_d0+i1*xstride_d1+i2*xstride_d2+0;
          t+=xslice[0];

          TYPE* rslice=r+i0*rstride_d0+i1*rstride_d1+i2*rstride_d2+0;
          for(int b0=0; b0<B0; b0++){
            rslice[b0*rstride_b0+0]+=t;
          }
        }
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D3_S0_B2(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int I1=params.ddims[1];
    int I2=params.ddims[2];
    int xstride_d0=params.xstride_d[0];
    int xstride_d1=params.xstride_d[1];
    int xstride_d2=params.xstride_d[2];
    int rstride_d0=params.rstride_d[0];
    int rstride_d1=params.rstride_d[1];
    int rstride_d2=params.rstride_d[2];
    int B0=params.sdims[0];
    int B1=params.sdims[1];
    int rstride_b0=params.rstride_b[0];
    int rstride_b1=params.rstride_b[1];

    for(int i0=0; i0<I0; i0++){
      for(int i1=0; i1<I1; i1++){
        for(int i2=0; i2<I2; i2++){

          TYPE t=0;
          const TYPE* xslice=x+i0*xstride_d0+i1*xstride_d1+i2*xstride_d2+0;
          t+=xslice[0];

          TYPE* rslice=r+i0*rstride_d0+i1*rstride_d1+i2*rstride_d2+0;
          for(int b0=0; b0<B0; b0++){
            for(int b1=0; b1<B1; b1++){
              rslice[b0*rstride_b0+b1*rstride_b1+0]+=t;
            }
          }
        }
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D3_S0_B3(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int I1=params.ddims[1];
    int I2=params.ddims[2];
    int xstride_d0=params.xstride_d[0];
    int xstride_d1=params.xstride_d[1];
    int xstride_d2=params.xstride_d[2];
    int rstride_d0=params.rstride_d[0];
    int rstride_d1=params.rstride_d[1];
    int rstride_d2=params.rstride_d[2];
    int B0=params.sdims[0];
    int B1=params.sdims[1];
    int B2=params.sdims[2];
    int rstride_b0=params.rstride_b[0];
    int rstride_b1=params.rstride_b[1];
    int rstride_b2=params.rstride_b[2];

    for(int i0=0; i0<I0; i0++){
      for(int i1=0; i1<I1; i1++){
        for(int i2=0; i2<I2; i2++){

          TYPE t=0;
          const TYPE* xslice=x+i0*xstride_d0+i1*xstride_d1+i2*xstride_d2+0;
          t+=xslice[0];

          TYPE* rslice=r+i0*rstride_d0+i1*rstride_d1+i2*rstride_d2+0;
          for(int b0=0; b0<B0; b0++){
            for(int b1=0; b1<B1; b1++){
              for(int b2=0; b2<B2; b2++){
                rslice[b0*rstride_b0+b1*rstride_b1+b2*rstride_b2+0]+=t;
              }
            }
          }
        }
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D3_S1_B0(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int I1=params.ddims[1];
    int I2=params.ddims[2];
    int xstride_d0=params.xstride_d[0];
    int xstride_d1=params.xstride_d[1];
    int xstride_d2=params.xstride_d[2];
    int rstride_d0=params.rstride_d[0];
    int rstride_d1=params.rstride_d[1];
    int rstride_d2=params.rstride_d[2];
    int S0=params.sdims[0];
    int xstride_s0=params.xstride_s[0];

    for(int i0=0; i0<I0; i0++){
      for(int i1=0; i1<I1; i1++){
        for(int i2=0; i2<I2; i2++){

          TYPE t=0;
          const TYPE* xslice=x+i0*xstride_d0+i1*xstride_d1+i2*xstride_d2+0;
          for(int s0=0; s0<S0; s0++){
            t+=xslice[s0*xstride_s0+0];
          }

          TYPE* rslice=r+i0*rstride_d0+i1*rstride_d1+i2*rstride_d2+0;
          rslice[0]+=t;
        }
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D3_S1_B1(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int I1=params.ddims[1];
    int I2=params.ddims[2];
    int xstride_d0=params.xstride_d[0];
    int xstride_d1=params.xstride_d[1];
    int xstride_d2=params.xstride_d[2];
    int rstride_d0=params.rstride_d[0];
    int rstride_d1=params.rstride_d[1];
    int rstride_d2=params.rstride_d[2];
    int S0=params.sdims[0];
    int xstride_s0=params.xstride_s[0];
    int B0=params.sdims[0];
    int rstride_b0=params.rstride_b[0];

    for(int i0=0; i0<I0; i0++){
      for(int i1=0; i1<I1; i1++){
        for(int i2=0; i2<I2; i2++){

          TYPE t=0;
          const TYPE* xslice=x+i0*xstride_d0+i1*xstride_d1+i2*xstride_d2+0;
          for(int s0=0; s0<S0; s0++){
            t+=xslice[s0*xstride_s0+0];
          }

          TYPE* rslice=r+i0*rstride_d0+i1*rstride_d1+i2*rstride_d2+0;
          for(int b0=0; b0<B0; b0++){
            rslice[b0*rstride_b0+0]+=t;
          }
        }
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D3_S1_B2(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int I1=params.ddims[1];
    int I2=params.ddims[2];
    int xstride_d0=params.xstride_d[0];
    int xstride_d1=params.xstride_d[1];
    int xstride_d2=params.xstride_d[2];
    int rstride_d0=params.rstride_d[0];
    int rstride_d1=params.rstride_d[1];
    int rstride_d2=params.rstride_d[2];
    int S0=params.sdims[0];
    int xstride_s0=params.xstride_s[0];
    int B0=params.sdims[0];
    int B1=params.sdims[1];
    int rstride_b0=params.rstride_b[0];
    int rstride_b1=params.rstride_b[1];

    for(int i0=0; i0<I0; i0++){
      for(int i1=0; i1<I1; i1++){
        for(int i2=0; i2<I2; i2++){

          TYPE t=0;
          const TYPE* xslice=x+i0*xstride_d0+i1*xstride_d1+i2*xstride_d2+0;
          for(int s0=0; s0<S0; s0++){
            t+=xslice[s0*xstride_s0+0];
          }

          TYPE* rslice=r+i0*rstride_d0+i1*rstride_d1+i2*rstride_d2+0;
          for(int b0=0; b0<B0; b0++){
            for(int b1=0; b1<B1; b1++){
              rslice[b0*rstride_b0+b1*rstride_b1+0]+=t;
            }
          }
        }
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D3_S1_B3(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int I1=params.ddims[1];
    int I2=params.ddims[2];
    int xstride_d0=params.xstride_d[0];
    int xstride_d1=params.xstride_d[1];
    int xstride_d2=params.xstride_d[2];
    int rstride_d0=params.rstride_d[0];
    int rstride_d1=params.rstride_d[1];
    int rstride_d2=params.rstride_d[2];
    int S0=params.sdims[0];
    int xstride_s0=params.xstride_s[0];
    int B0=params.sdims[0];
    int B1=params.sdims[1];
    int B2=params.sdims[2];
    int rstride_b0=params.rstride_b[0];
    int rstride_b1=params.rstride_b[1];
    int rstride_b2=params.rstride_b[2];

    for(int i0=0; i0<I0; i0++){
      for(int i1=0; i1<I1; i1++){
        for(int i2=0; i2<I2; i2++){

          TYPE t=0;
          const TYPE* xslice=x+i0*xstride_d0+i1*xstride_d1+i2*xstride_d2+0;
          for(int s0=0; s0<S0; s0++){
            t+=xslice[s0*xstride_s0+0];
          }

          TYPE* rslice=r+i0*rstride_d0+i1*rstride_d1+i2*rstride_d2+0;
          for(int b0=0; b0<B0; b0++){
            for(int b1=0; b1<B1; b1++){
              for(int b2=0; b2<B2; b2++){
                rslice[b0*rstride_b0+b1*rstride_b1+b2*rstride_b2+0]+=t;
              }
            }
          }
        }
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D3_S2_B0(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int I1=params.ddims[1];
    int I2=params.ddims[2];
    int xstride_d0=params.xstride_d[0];
    int xstride_d1=params.xstride_d[1];
    int xstride_d2=params.xstride_d[2];
    int rstride_d0=params.rstride_d[0];
    int rstride_d1=params.rstride_d[1];
    int rstride_d2=params.rstride_d[2];
    int S0=params.sdims[0];
    int S1=params.sdims[1];
    int xstride_s0=params.xstride_s[0];
    int xstride_s1=params.xstride_s[1];

    for(int i0=0; i0<I0; i0++){
      for(int i1=0; i1<I1; i1++){
        for(int i2=0; i2<I2; i2++){

          TYPE t=0;
          const TYPE* xslice=x+i0*xstride_d0+i1*xstride_d1+i2*xstride_d2+0;
          for(int s0=0; s0<S0; s0++){
            for(int s1=0; s1<S1; s1++){
              t+=xslice[s0*xstride_s0+s1*xstride_s1+0];
            }
          }

          TYPE* rslice=r+i0*rstride_d0+i1*rstride_d1+i2*rstride_d2+0;
          rslice[0]+=t;
        }
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D3_S2_B1(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int I1=params.ddims[1];
    int I2=params.ddims[2];
    int xstride_d0=params.xstride_d[0];
    int xstride_d1=params.xstride_d[1];
    int xstride_d2=params.xstride_d[2];
    int rstride_d0=params.rstride_d[0];
    int rstride_d1=params.rstride_d[1];
    int rstride_d2=params.rstride_d[2];
    int S0=params.sdims[0];
    int S1=params.sdims[1];
    int xstride_s0=params.xstride_s[0];
    int xstride_s1=params.xstride_s[1];
    int B0=params.sdims[0];
    int rstride_b0=params.rstride_b[0];

    for(int i0=0; i0<I0; i0++){
      for(int i1=0; i1<I1; i1++){
        for(int i2=0; i2<I2; i2++){

          TYPE t=0;
          const TYPE* xslice=x+i0*xstride_d0+i1*xstride_d1+i2*xstride_d2+0;
          for(int s0=0; s0<S0; s0++){
            for(int s1=0; s1<S1; s1++){
              t+=xslice[s0*xstride_s0+s1*xstride_s1+0];
            }
          }

          TYPE* rslice=r+i0*rstride_d0+i1*rstride_d1+i2*rstride_d2+0;
          for(int b0=0; b0<B0; b0++){
            rslice[b0*rstride_b0+0]+=t;
          }
        }
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D3_S2_B2(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int I1=params.ddims[1];
    int I2=params.ddims[2];
    int xstride_d0=params.xstride_d[0];
    int xstride_d1=params.xstride_d[1];
    int xstride_d2=params.xstride_d[2];
    int rstride_d0=params.rstride_d[0];
    int rstride_d1=params.rstride_d[1];
    int rstride_d2=params.rstride_d[2];
    int S0=params.sdims[0];
    int S1=params.sdims[1];
    int xstride_s0=params.xstride_s[0];
    int xstride_s1=params.xstride_s[1];
    int B0=params.sdims[0];
    int B1=params.sdims[1];
    int rstride_b0=params.rstride_b[0];
    int rstride_b1=params.rstride_b[1];

    for(int i0=0; i0<I0; i0++){
      for(int i1=0; i1<I1; i1++){
        for(int i2=0; i2<I2; i2++){

          TYPE t=0;
          const TYPE* xslice=x+i0*xstride_d0+i1*xstride_d1+i2*xstride_d2+0;
          for(int s0=0; s0<S0; s0++){
            for(int s1=0; s1<S1; s1++){
              t+=xslice[s0*xstride_s0+s1*xstride_s1+0];
            }
          }

          TYPE* rslice=r+i0*rstride_d0+i1*rstride_d1+i2*rstride_d2+0;
          for(int b0=0; b0<B0; b0++){
            for(int b1=0; b1<B1; b1++){
              rslice[b0*rstride_b0+b1*rstride_b1+0]+=t;
            }
          }
        }
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D3_S2_B3(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int I1=params.ddims[1];
    int I2=params.ddims[2];
    int xstride_d0=params.xstride_d[0];
    int xstride_d1=params.xstride_d[1];
    int xstride_d2=params.xstride_d[2];
    int rstride_d0=params.rstride_d[0];
    int rstride_d1=params.rstride_d[1];
    int rstride_d2=params.rstride_d[2];
    int S0=params.sdims[0];
    int S1=params.sdims[1];
    int xstride_s0=params.xstride_s[0];
    int xstride_s1=params.xstride_s[1];
    int B0=params.sdims[0];
    int B1=params.sdims[1];
    int B2=params.sdims[2];
    int rstride_b0=params.rstride_b[0];
    int rstride_b1=params.rstride_b[1];
    int rstride_b2=params.rstride_b[2];

    for(int i0=0; i0<I0; i0++){
      for(int i1=0; i1<I1; i1++){
        for(int i2=0; i2<I2; i2++){

          TYPE t=0;
          const TYPE* xslice=x+i0*xstride_d0+i1*xstride_d1+i2*xstride_d2+0;
          for(int s0=0; s0<S0; s0++){
            for(int s1=0; s1<S1; s1++){
              t+=xslice[s0*xstride_s0+s1*xstride_s1+0];
            }
          }

          TYPE* rslice=r+i0*rstride_d0+i1*rstride_d1+i2*rstride_d2+0;
          for(int b0=0; b0<B0; b0++){
            for(int b1=0; b1<B1; b1++){
              for(int b2=0; b2<B2; b2++){
                rslice[b0*rstride_b0+b1*rstride_b1+b2*rstride_b2+0]+=t;
              }
            }
          }
        }
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D3_S3_B0(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int I1=params.ddims[1];
    int I2=params.ddims[2];
    int xstride_d0=params.xstride_d[0];
    int xstride_d1=params.xstride_d[1];
    int xstride_d2=params.xstride_d[2];
    int rstride_d0=params.rstride_d[0];
    int rstride_d1=params.rstride_d[1];
    int rstride_d2=params.rstride_d[2];
    int S0=params.sdims[0];
    int S1=params.sdims[1];
    int S2=params.sdims[2];
    int xstride_s0=params.xstride_s[0];
    int xstride_s1=params.xstride_s[1];
    int xstride_s2=params.xstride_s[2];

    for(int i0=0; i0<I0; i0++){
      for(int i1=0; i1<I1; i1++){
        for(int i2=0; i2<I2; i2++){

          TYPE t=0;
          const TYPE* xslice=x+i0*xstride_d0+i1*xstride_d1+i2*xstride_d2+0;
          for(int s0=0; s0<S0; s0++){
            for(int s1=0; s1<S1; s1++){
              for(int s2=0; s2<S2; s2++){
                t+=xslice[s0*xstride_s0+s1*xstride_s1+s2*xstride_s2+0];
              }
            }
          }

          TYPE* rslice=r+i0*rstride_d0+i1*rstride_d1+i2*rstride_d2+0;
          rslice[0]+=t;
        }
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D3_S3_B1(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int I1=params.ddims[1];
    int I2=params.ddims[2];
    int xstride_d0=params.xstride_d[0];
    int xstride_d1=params.xstride_d[1];
    int xstride_d2=params.xstride_d[2];
    int rstride_d0=params.rstride_d[0];
    int rstride_d1=params.rstride_d[1];
    int rstride_d2=params.rstride_d[2];
    int S0=params.sdims[0];
    int S1=params.sdims[1];
    int S2=params.sdims[2];
    int xstride_s0=params.xstride_s[0];
    int xstride_s1=params.xstride_s[1];
    int xstride_s2=params.xstride_s[2];
    int B0=params.sdims[0];
    int rstride_b0=params.rstride_b[0];

    for(int i0=0; i0<I0; i0++){
      for(int i1=0; i1<I1; i1++){
        for(int i2=0; i2<I2; i2++){

          TYPE t=0;
          const TYPE* xslice=x+i0*xstride_d0+i1*xstride_d1+i2*xstride_d2+0;
          for(int s0=0; s0<S0; s0++){
            for(int s1=0; s1<S1; s1++){
              for(int s2=0; s2<S2; s2++){
                t+=xslice[s0*xstride_s0+s1*xstride_s1+s2*xstride_s2+0];
              }
            }
          }

          TYPE* rslice=r+i0*rstride_d0+i1*rstride_d1+i2*rstride_d2+0;
          for(int b0=0; b0<B0; b0++){
            rslice[b0*rstride_b0+0]+=t;
          }
        }
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D3_S3_B2(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int I1=params.ddims[1];
    int I2=params.ddims[2];
    int xstride_d0=params.xstride_d[0];
    int xstride_d1=params.xstride_d[1];
    int xstride_d2=params.xstride_d[2];
    int rstride_d0=params.rstride_d[0];
    int rstride_d1=params.rstride_d[1];
    int rstride_d2=params.rstride_d[2];
    int S0=params.sdims[0];
    int S1=params.sdims[1];
    int S2=params.sdims[2];
    int xstride_s0=params.xstride_s[0];
    int xstride_s1=params.xstride_s[1];
    int xstride_s2=params.xstride_s[2];
    int B0=params.sdims[0];
    int B1=params.sdims[1];
    int rstride_b0=params.rstride_b[0];
    int rstride_b1=params.rstride_b[1];

    for(int i0=0; i0<I0; i0++){
      for(int i1=0; i1<I1; i1++){
        for(int i2=0; i2<I2; i2++){

          TYPE t=0;
          const TYPE* xslice=x+i0*xstride_d0+i1*xstride_d1+i2*xstride_d2+0;
          for(int s0=0; s0<S0; s0++){
            for(int s1=0; s1<S1; s1++){
              for(int s2=0; s2<S2; s2++){
                t+=xslice[s0*xstride_s0+s1*xstride_s1+s2*xstride_s2+0];
              }
            }
          }

          TYPE* rslice=r+i0*rstride_d0+i1*rstride_d1+i2*rstride_d2+0;
          for(int b0=0; b0<B0; b0++){
            for(int b1=0; b1<B1; b1++){
              rslice[b0*rstride_b0+b1*rstride_b1+0]+=t;
            }
          }
        }
      }
    }

  }


  template<typename TYPE>
  void Einsum1_D3_S3_B3(TYPE* r, const TYPE* x, const EsumParams& params){

    int I0=params.ddims[0];
    int I1=params.ddims[1];
    int I2=params.ddims[2];
    int xstride_d0=params.xstride_d[0];
    int xstride_d1=params.xstride_d[1];
    int xstride_d2=params.xstride_d[2];
    int rstride_d0=params.rstride_d[0];
    int rstride_d1=params.rstride_d[1];
    int rstride_d2=params.rstride_d[2];
    int S0=params.sdims[0];
    int S1=params.sdims[1];
    int S2=params.sdims[2];
    int xstride_s0=params.xstride_s[0];
    int xstride_s1=params.xstride_s[1];
    int xstride_s2=params.xstride_s[2];
    int B0=params.sdims[0];
    int B1=params.sdims[1];
    int B2=params.sdims[2];
    int rstride_b0=params.rstride_b[0];
    int rstride_b1=params.rstride_b[1];
    int rstride_b2=params.rstride_b[2];

    for(int i0=0; i0<I0; i0++){
      for(int i1=0; i1<I1; i1++){
        for(int i2=0; i2<I2; i2++){

          TYPE t=0;
          const TYPE* xslice=x+i0*xstride_d0+i1*xstride_d1+i2*xstride_d2+0;
          for(int s0=0; s0<S0; s0++){
            for(int s1=0; s1<S1; s1++){
              for(int s2=0; s2<S2; s2++){
                t+=xslice[s0*xstride_s0+s1*xstride_s1+s2*xstride_s2+0];
              }
            }
          }

          TYPE* rslice=r+i0*rstride_d0+i1*rstride_d1+i2*rstride_d2+0;
          for(int b0=0; b0<B0; b0++){
            for(int b1=0; b1<B1; b1++){
              for(int b2=0; b2<B2; b2++){
                rslice[b0*rstride_b0+b1*rstride_b1+b2*rstride_b2+0]+=t;
              }
            }
          }
        }
      }
    }

  }




  void LtensorEinsum1loops(int D, int S, int B, float* r, const float* x, const EsumParams& params){
    switch(D){
    case 0:
      switch(S){
      case 0:
        switch(B){
        case 0:
          Einsum1_D0_S0_B0(r,x,params);
          break;
        case 1:
          Einsum1_D0_S0_B1(r,x,params);
          break;
        case 2:
          Einsum1_D0_S0_B2(r,x,params);
          break;
        case 3:
          Einsum1_D0_S0_B3(r,x,params);
          break;
        }
        break;
      case 1:
        switch(B){
        case 0:
          Einsum1_D0_S1_B0(r,x,params);
          break;
        case 1:
          Einsum1_D0_S1_B1(r,x,params);
          break;
        case 2:
          Einsum1_D0_S1_B2(r,x,params);
          break;
        case 3:
          Einsum1_D0_S1_B3(r,x,params);
          break;
        }
        break;
      case 2:
        switch(B){
        case 0:
          Einsum1_D0_S2_B0(r,x,params);
          break;
        case 1:
          Einsum1_D0_S2_B1(r,x,params);
          break;
        case 2:
          Einsum1_D0_S2_B2(r,x,params);
          break;
        case 3:
          Einsum1_D0_S2_B3(r,x,params);
          break;
        }
        break;
      case 3:
        switch(B){
        case 0:
          Einsum1_D0_S3_B0(r,x,params);
          break;
        case 1:
          Einsum1_D0_S3_B1(r,x,params);
          break;
        case 2:
          Einsum1_D0_S3_B2(r,x,params);
          break;
        case 3:
          Einsum1_D0_S3_B3(r,x,params);
          break;
        }
        break;
      }
      break;
    case 1:
      switch(S){
      case 0:
        switch(B){
        case 0:
          Einsum1_D1_S0_B0(r,x,params);
          break;
        case 1:
          Einsum1_D1_S0_B1(r,x,params);
          break;
        case 2:
          Einsum1_D1_S0_B2(r,x,params);
          break;
        case 3:
          Einsum1_D1_S0_B3(r,x,params);
          break;
        }
        break;
      case 1:
        switch(B){
        case 0:
          Einsum1_D1_S1_B0(r,x,params);
          break;
        case 1:
          Einsum1_D1_S1_B1(r,x,params);
          break;
        case 2:
          Einsum1_D1_S1_B2(r,x,params);
          break;
        case 3:
          Einsum1_D1_S1_B3(r,x,params);
          break;
        }
        break;
      case 2:
        switch(B){
        case 0:
          Einsum1_D1_S2_B0(r,x,params);
          break;
        case 1:
          Einsum1_D1_S2_B1(r,x,params);
          break;
        case 2:
          Einsum1_D1_S2_B2(r,x,params);
          break;
        case 3:
          Einsum1_D1_S2_B3(r,x,params);
          break;
        }
        break;
      case 3:
        switch(B){
        case 0:
          Einsum1_D1_S3_B0(r,x,params);
          break;
        case 1:
          Einsum1_D1_S3_B1(r,x,params);
          break;
        case 2:
          Einsum1_D1_S3_B2(r,x,params);
          break;
        case 3:
          Einsum1_D1_S3_B3(r,x,params);
          break;
        }
        break;
      }
      break;
    case 2:
      switch(S){
      case 0:
        switch(B){
        case 0:
          Einsum1_D2_S0_B0(r,x,params);
          break;
        case 1:
          Einsum1_D2_S0_B1(r,x,params);
          break;
        case 2:
          Einsum1_D2_S0_B2(r,x,params);
          break;
        case 3:
          Einsum1_D2_S0_B3(r,x,params);
          break;
        }
        break;
      case 1:
        switch(B){
        case 0:
          Einsum1_D2_S1_B0(r,x,params);
          break;
        case 1:
          Einsum1_D2_S1_B1(r,x,params);
          break;
        case 2:
          Einsum1_D2_S1_B2(r,x,params);
          break;
        case 3:
          Einsum1_D2_S1_B3(r,x,params);
          break;
        }
        break;
      case 2:
        switch(B){
        case 0:
          Einsum1_D2_S2_B0(r,x,params);
          break;
        case 1:
          Einsum1_D2_S2_B1(r,x,params);
          break;
        case 2:
          Einsum1_D2_S2_B2(r,x,params);
          break;
        case 3:
          Einsum1_D2_S2_B3(r,x,params);
          break;
        }
        break;
      case 3:
        switch(B){
        case 0:
          Einsum1_D2_S3_B0(r,x,params);
          break;
        case 1:
          Einsum1_D2_S3_B1(r,x,params);
          break;
        case 2:
          Einsum1_D2_S3_B2(r,x,params);
          break;
        case 3:
          Einsum1_D2_S3_B3(r,x,params);
          break;
        }
        break;
      }
      break;
    case 3:
      switch(S){
      case 0:
        switch(B){
        case 0:
          Einsum1_D3_S0_B0(r,x,params);
          break;
        case 1:
          Einsum1_D3_S0_B1(r,x,params);
          break;
        case 2:
          Einsum1_D3_S0_B2(r,x,params);
          break;
        case 3:
          Einsum1_D3_S0_B3(r,x,params);
          break;
        }
        break;
      case 1:
        switch(B){
        case 0:
          Einsum1_D3_S1_B0(r,x,params);
          break;
        case 1:
          Einsum1_D3_S1_B1(r,x,params);
          break;
        case 2:
          Einsum1_D3_S1_B2(r,x,params);
          break;
        case 3:
          Einsum1_D3_S1_B3(r,x,params);
          break;
        }
        break;
      case 2:
        switch(B){
        case 0:
          Einsum1_D3_S2_B0(r,x,params);
          break;
        case 1:
          Einsum1_D3_S2_B1(r,x,params);
          break;
        case 2:
          Einsum1_D3_S2_B2(r,x,params);
          break;
        case 3:
          Einsum1_D3_S2_B3(r,x,params);
          break;
        }
        break;
      case 3:
        switch(B){
        case 0:
          Einsum1_D3_S3_B0(r,x,params);
          break;
        case 1:
          Einsum1_D3_S3_B1(r,x,params);
          break;
        case 2:
          Einsum1_D3_S3_B2(r,x,params);
          break;
        case 3:
          Einsum1_D3_S3_B3(r,x,params);
          break;
        }
        break;
      }
      break;
    }
  }


}
