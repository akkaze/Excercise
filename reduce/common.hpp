#ifndef COMMON_HPP
#define COMMON_HPP

#include <iostream>
__host__ __forceinline__ void checkCudaError(cudaError_t err, const char* file,const char* func)
{
    if(cudaSuccess != err)
            std::cerr << cudaGetErrorString(err) << '\n';
}

__host__ __device__ __forceinline__ int divUp(int total,int grain)
{
    return (total + grain - 1) / grain;
}
#endif