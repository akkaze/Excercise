#include <assert.h>

inline bool isPowerOfTwo(int n)
{
    return ((n& (n-1) ) == 0);
}

inline int floorPow2(int n)
{
    int exp;
    frexp((float)n,&exp);
    return 1 << (exp - 1);
}

#define BLOCK_SIZE 256

float** scan_block_sums;
unsigned int num_elts_allocated = 0;
unsigned int num_lvls_allocated = 0;

void preallocBlockSums(unsigned max_num_elts)
{
    assert(num_elts_allocated == 0);
    num_elts_allocated = max_num_elts;
    unsigned int block_size = BLOCK_SIZE;
    unsigned int num_elts = max_num_elts;

    int lvl = 0;

    do
    {
        unsigned int num_blocks = 
            max(1,(int)ceil((float)num_elts / (2.f * block_size)));
        if(num_blocks > 1)
            lvl++;
    } while (num_elts > 1); 

    scan_block_sums = (float**)malloc(lvl * sizeof(float*));
    num_lvls_allocated = lvl;

    num_elts = max_num_elts;
    lvl = 0;

    do
    {
        unsigned int num_blocks = 
            max(1,(int)ceil((float)num_elts / (2.f * block_size)));
        if(num_blocks > 1)
            cudaSafeCall(cudaMalloc((void**)&scan_block_sums[lvl++], num_blocks * sizeof(float)));
        num_elts = num_blocks;
    }while(num_elts > 1);
}

void releaseBlockSums()
{
    for(int i = 0; i < num_lvls_allocated; ++i)
        cudaFree(scan_block_sums[i]);

    free((void**)scan_block_sums);

    scan_block_sums = NULL;
    num_elts_allocated = NULL;
    num_lvls_allocated = NULL;
}

void prescanArrayRecursive(float* out,const float* in,int num_elts,int lvl)
{
    unsigned int block_size = BLOCK_SIZE;
    unsigned int num_blocks = 
        max(1,(int)ceil((float)num_elts / (2.f * block_size)));
    unsigned int num_threads;

    if(num_blocks > 1)
        num_threads = block_size;
    else if(isPowerOfTwo(num_elts))
        num_threads = num_elts / 2;
    else    
        num_threads = floorPow2(num_elts);

    unsigned int num_elts_per_block = num_threads * 2;

    unsigned int num_elts_last_block = 
        num_elts - (num_blocks - 1) * num_elts_per_block;
    unsigned int num_thread_last_block = 
        max(1, num_elts_last_block / 2);
    unsigned int np2_last_block = 0;
    unsigned int smem_last_block = 0;

    if(num_elts_last_block != num_elts_per_block)
    {
        np2_last_block = 1;

        if(!isPowerOfTwo(num_elts_last_block))
            num_thread_last_block = floorPow2(num_elts_last_block);
        unsigned int extra_space = (2 * num_thread_last_block) / NUM_BANKS;
        smem_last_block = sizeof(float) * (2 * num_thread_last_block + extra_space);
    }

    unsigned int extra_space = num_elts_per_block / NUM_BANKS;
    unsigned int smem_size = sizeof(float) * (num_elts_per_block + extra_space);

#ifdef DEBUG
    if(num_blocks > 1)
        assert(num_elts_allocated >= num_elts);
#endif

    dim3 gird(max(1,num_blocks - np2_last_block),1,1);
    dim3 block(num_threads,1,1);

    if(num_blocks > 1)
    {
        
    }
}