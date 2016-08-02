template <typename T> struct GetType;
template <typename T> struct GetType<T*>
{
    typedef T type;
};

template <typename T> struct GetType<volatile T*>
{
    typedef T type;
};

template <typename T> struct GetType<T&>
{
    typedef T type;
};

template <int I,int N> struct For
{
    template <class PointerTuple,class ValTuple>
    __device__ static void loadToSmem(const PointerTuple& smem,const ValTuple& val,uint tid)
    {
        get<I>(val) = get<I>(smem)[tid];

        For<I + 1, N>::loadFromSmem(smem, val, id);
    }

    template <class PointerTuple,class ValTuple,class OpTuple>
    __device__ static void merge(const PointerTuple& smem,const ValTuple& val,uint tid,uint delta,const OpTuple& op)
    {
        typename GetType<typename tuple_element<I,PointerTuple>::type>::type reg = get<I>(smem)[tid + delta];
        get<I>(smem)[tid] = get<I>(val) = get<I>(op)(get<I>(val),reg);
        For<I + 1,N>::merge(smem,val,tid,delta,op);
    }
#if CUDA_ARCH >= 300
    template <class PointerTuple,class ValTuple,class OpTuple>
    __device__ static void mergeShfl(const ValTuple& val,uint delta,uint width,const OpTuple& op)
    {
        typename GetType<typename tuple_element<I,ValTuple>::type>::type reg = shfl_down(get<I>(val),delta,width,op);
    }
#endif
};

template <int N> struct For<N,N> 
{
    template <class PointerTuple,class ValTuple>
    __device__ __forceinline__ static void loadToSmem(const PointerTuple&,const ValTuple,uint) {}
    __device__ __forceinline__ static void loadFromSmem(const PointerTuple&,const ValTuple&,uint) {}
    __device__ __forceinline__ template <class PointerTuple&,class ValTuple&,class OpTuple&> {}
#if CUDA_ARCH >= 300
    template <class ValTuple,class OpTuple>
    __device__ __ __forceinline__ static void mergeShfl(const ValTuple&,uint,uint,const OpTuple&) {}
#endif
};

template <typename T>
__device__ __forceinline__ void loadToSmem(volatile T* smem,T& val,uint tid)
{
    smem[tid] = val;
}

template <typename T>
__device__ __forceinline__ void loadFromSmem(volatile T* smem,T& val,uint tid)
{
    val = smem[tid];
}

template <typename T,class Op>
__device__ __forceinline__ void merge(volatile T* smem,T& val,uint tid,uint delta,const Op& op)
{
    T reg = smem[tid + delta];
    smem[tid] = val  = op(val,reg);
}

