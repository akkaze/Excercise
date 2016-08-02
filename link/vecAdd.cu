#include "Test.h"

__global__ void vecAdd(float *a, float *b, float *c, int n) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;

        if(i < n)
        {
        for(int j=0;j<1000000;++j)
                c[i] = a[i] / b[i];
				}
}

void Test::addVec_gpu(float *a, float *b, float *c, int n) {
        float *d_a, *d_b, *d_c;
        size_t size = n * sizeof(float);

        cudaMalloc(&d_a, size);
        cudaMalloc(&d_b, size);
        cudaMalloc(&d_c, size);
				
				cudaMemcpy(d_a, a, n*sizeof(float), cudaMemcpyHostToDevice);
				cudaMemcpy(d_b, b, n*sizeof(float), cudaMemcpyHostToDevice);
				cudaMemcpy(d_c, c, n*sizeof(float), cudaMemcpyHostToDevice);
				vecAdd<<<(n/256)+1,256>>>(d_a,d_b,d_c,n);
				cudaMemcpy(a, d_a, n*sizeof(float), cudaMemcpyDeviceToHost);
				cudaMemcpy(b, d_b, n*sizeof(float), cudaMemcpyDeviceToHost);
				cudaMemcpy(c, d_c, n*sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
}


