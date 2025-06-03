
namespace cnine{

  template<class TYPE, typename... Args>
  vector<TYPE*> variadic_unroller(TYPE& x, Args&... args){
    vector<TYPE*> argv;
    variadic_unroller_sub(argv, x, args...);
    return argv;}

  template<class TYPE, typename... Args>
  void variadic_unroller_sub(vector<TYPE*>& argv, TYPE& x, Args&... args){
    argv.push_back(&x);
    variadic_unroller_sub(argv, args...);}

  template<class TYPE, typename... Args>
  void variadic_unroller_sub(vector<TYPE*>& argv, TYPE& x){
    argv.push_back(&x);}


  template<class TYPE, typename... Args>
  void const_variadic_unroller_sub(vector<const TYPE*>& argv, const TYPE& x){
    argv.push_back(&x);}

  template<class TYPE, typename... Args>
  void const_variadic_unroller_sub(vector<const TYPE*>& argv, const TYPE& x, Args&... args){
    argv.push_back(&x);
    const_variadic_unroller_sub(argv, args...);}

  template<class TYPE, typename... Args>
  vector<const TYPE*> const_variadic_unroller(const TYPE& x, Args&... args){
    vector<const TYPE*> argv;
    const_variadic_unroller_sub(argv, x, args...);
    return argv;}


  template<class TYPE, class TYPE2, typename... Args>
  void const_derived_variadic_unroller_sub(vector<const TYPE*>& argv, const TYPE2& x, Args&... args){
    argv.push_back(&x);
    const_derived_variadic_unroller_sub(argv, args...);
  }

  template<class TYPE, class TYPE2> 
  void const_derived_variadic_unroller_sub(vector<const TYPE*>& argv, const TYPE2& x){
    argv.push_back(&x);
  }

  template<class TYPE, typename... Args>
  vector<const TYPE*> const_derived_variadic_unroller(Args&... args){
    vector<const TYPE*> argv;
    const_derived_variadic_unroller_sub<TYPE>(argv, args...);
    return argv;
  }

}
