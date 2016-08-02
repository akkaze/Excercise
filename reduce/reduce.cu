#include "common.hpp"
#define BLOCK_SIZE 512

__global__ void sum(float* input,float* output,int len)
{
    __shared__ float partial_sum[2 * BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * blockDim.x;

    partial_sum[tid] = (tid < len) ? input[start + tid] : 0;
    partial_sum[blockDim.x + tid] = ((blockDim.x + tid) < len) ? input[start + blockDim.x + tid] : 0;

    for(unsigned int stride = blockDim.x; stride >= 1; stride >>= 1)
    {
        __syncthreads();
        if(tid < stride)
            partial_sum[t] = partial_sum[t + stride];
    }

    if(tid == 0)
    {
        output[blockIdx.x + tid] = partial_sum[tid];
    }
}

int main(int argc, char** argv)
{
    int idx;
    float* host_input;
    float* host_output;
    float* device_input;
    float* device_output;
    int num_input;
    int num_output;

    num_output = num_input / (BLOCK_SIZE << 1);
    if(num_input % (BLOCK_SIZE << 1))
        num_output++;
    host_output = (float*)malloc(num_output * sizeof(float));
    cudaMalloc((void**)&device_input,num_input * sizeof(float));
    cudaMalloc((void**)&device_output,num_output * sizeof(float));
    cudaMemcpy(device_input,host_input,cudaMemHostToDevice);
    dim3 grid((num_input - 1) / BLOCK_SIZE + 1,1,1);
    dim3 block(BLOCK_SIZE,1,1);

    sum<<<grid,block>>>(device_input,device_output,num_input);

    cudaDeviceSynchronize();

    cudaMemcpy(host_output,device_output,cudaMemDeviceToHost);

    for(idx = 1; idx < num_output; idx++)
        host_output[0] += host_output[idx];

    cudaFree(device_input);
    cudaFree(device_output);

    free(host_input);
    free(host_output);
    cudaMemc
}