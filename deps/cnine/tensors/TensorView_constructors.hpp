

public: // ---- Named parameter constructors ---------------------------------------------------------------


struct vparams{
  Gdims dims;
  int fcode=0;
  int dev=0;
};      

template<typename ARG0, typename... Args, 
	     typename = typename std::enable_if<
    std::is_same<DimsArgument, ARG0>::value ||
    std::is_same<FillArgument, ARG0>::value ||
    std::is_same<DeviceArgument, ARG0>::value, ARG0>::type>
TensorView(const ARG0& arg0, const Args&... args){
  vparams v;
  unroller(v,arg0,args...);
  reset(TensorView(v.dims,v.fcode,v.dev));
}

template<typename... Args>
void unroller(vparams& v, const cnine::DimsArgument& x, const Args&... args){
  v.dims=x.get(); unroller(v, args...);}

template<typename... Args>
void unroller(vparams& v, const cnine::FillArgument& x, const Args&... args){
  v.fcode=x.get(); unroller(v, args...);}

template<typename... Args>
void unroller(vparams& v, const cnine::DeviceArgument& x, const Args&... args){
  v.dev=x.get(); unroller(v, args...);}

void unroller(vparams& v){}



public: // ---- Fill constructors ---------------------------------------------------------------------------------


//[[deprecated]] // avoid because second argument is easily confused
TensorView(const Gdims& _dims, const int _dev=0): 
  TensorView(_dims,0,_dev){}

TensorView(const Gdims& _dims, const fill_zero& dummy, const int _dev=0): 
  TensorView(MemArr<TYPE>(_dims.asize(),dummy,_dev),_dims,GstridesB(_dims)){
}

TensorView(const Gdims& _dims, const fill_raw& dummy, const int _dev=0): 
  TensorView(MemArr<TYPE>(_dims.asize(),_dev),_dims,GstridesB(_dims)){}


// TODO 
TensorView(const Gdims& _dims, const fill_constant<TYPE>& dummy, const int _dev=0):
  TensorView(_dims,0){
  size_t N=dims.asize();
  for(size_t i=0; i<N; i++)
    arr[i]=dummy.v;
  move_to_device(_dev);
}

// TODO 
TensorView(const Gdims& _dims, const fill_identity& dummy, const int _dev=0):
  TensorView(_dims,fill_zero(),0){
  CNINE_ASSRT(ndims()==2);
  CNINE_ASSRT(dim(0)==dim(1));
  int N=dim(0);
  for(int i=0; i<N; i++)
    set(i,i,1.0);
  move_to_device(_dev);
}

TensorView(const Gdims& _dims, const fill_sequential& dummy, const int _dev=0):
  TensorView(_dims,0){
  size_t N=dims.asize();
  for(size_t i=0; i<N; i++)
    arr[i]=i;
  move_to_device(_dev);
}

TensorView(const Gdims& _dims, const fill_gaussian& dummy, const int _dev=0):
  TensorView(_dims,0){
  int N=dims.asize();
  if constexpr(is_complex<TYPE>()){
    normal_distribution<TYPE> distr;
    for(int i=0; i<N; i++) 
      arr[i]=TYPE(distr(rndGen),distr(rndGen))*dummy.c;
  }else{
    normal_distribution<TYPE> distr;
    for(int i=0; i<N; i++) 
      arr[i]=distr(rndGen)*dummy.c;
  }
  move_to_device(_dev);
}


public: // ---- Named constructors ------------------------------------------------------------------------


static TensorView zero(const Gdims& _dims, const int _dev=0){
  return TensorView(_dims,0,_dev);
}

static TensorView raw(const Gdims& _dims, const int _dev=0){
  return TensorView(_dims,1,_dev);
}

static TensorView ones(const Gdims& _dims, const int _dev=0){
  return TensorView(_dims,2,_dev);
}

static TensorView sequential(const Gdims& _dims, const int _dev=0){
  return TensorView(_dims,3,_dev);
}

static TensorView gaussian(const Gdims& _dims, const int _dev=0){
  return TensorView(_dims,4,_dev);
}

static TensorView identity(const int n, const int _dev=0){
  return TensorView({n,n},fill_identity(),_dev);
}

static TensorView identity(const Gdims& _dims, const int _dev=0){
  return TensorView(_dims,fill_identity(),_dev);
}

static TensorView unit(const int n, const int i, const int _dev=0){
  TensorView<TYPE> R({n},0,0);
  R.set(i,1);
  R.move_to_device(_dev);
  return R;
}

static TensorView random_unitary(const Gdims& _dims, const int _dev=0){
  TensorView R(_dims,0,0);
  CNINE_ASSRT(R.ndims()==2);
  CNINE_ASSRT(R.dim(0)==R.dim(1));
  int N=R.dim(0);
  for(int i=0; i<N; i++){
    auto v=TensorView({N},4,0);
    for(int j=0; j<i; j++){
      auto u=R.row(j); 
      v.subtract(u*u.inp(v));
    }
    R.row(i).add(v,TYPE(1.0)/v.norm());
  }
  R.move_to_device(_dev);
  return R;
}


public: // ---- Like constructors ----------------------------------------------------------------------------


TensorView zeros_like() const{
  return TensorView(dims,0,dev);
}

TensorView ones_like() const{
  return TensorView(dims,2,dev);
}

TensorView gaussian_like() const{
  return TensorView(dims,4,dev);
}


public: // ---- Initializing constructors ---------------------------------------------------------------------


static TensorView vec(const initializer_list<TYPE>& list, const int _dev=0){
  int n0=list.size();
  CNINE_ASSRT(n0>0);
  TensorView<TYPE> T(Gdims({n0}),0,0); 
  int i=0;
  for(auto& p: list){
    T.set(i,p);
    i++;
  }
  if(_dev>0) T.move_to_device(_dev);
  return T;
}


TensorView(const initializer_list<initializer_list<TYPE> >& list, const int _dev=0){
  int n0=list.size();
  CNINE_ASSRT(n0>0);
  int n1=list.begin()->size();
  TensorView<TYPE> T(Gdims({n0,n1}),0,0); 
  int i=0;
  for(auto& p: list){
    int j=0;
    for(auto& q: p)
      T.set(i,j++,q);
    i++;
  }
  if(_dev>0) T.move_to_device(_dev);
  reset(T);
}



// TODO 
public: // ---- Stacking ----------------------------------------------------------------------------------


template<typename OBJ>
static TensorView stack(int d, const vector<OBJ>& list){
  CNINE_ASSRT(list.size()>0);
  CNINE_ASSRT(d<list[0].ndims());
  Gdims dims0=list[0].dims;
  Gdims rem=list[0].dims.remove(d);
  int t=0;
  for(int i=0; i<list.size(); i++){
    t+=list[i].dim(d);
    CNINE_ASSRT(list[i].dims.remove(d)==rem);
  }
  TensorView R(dims0.set(d,t),0,list[0].get_dev());
  t=0;
  for(int i=0; i<list.size(); i++){
    R.slices(d,t,list[i].dim(d))+=list[i];
    t+=list[i].dim(d);
  }
  return R;
}

template<typename OBJ>
static TensorView stack(int d, const initializer_list<OBJ>& list){
  vector<TensorView<TYPE> > x;
  for(auto& p:list) x.push_back(p);
  return stack(0,x);
}



