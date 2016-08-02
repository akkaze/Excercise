#pragma once 
#include <cstddef>

//#define CNN_USE_TBB

//#define CNN_USE_AVX

//#define CNN_USE_SSE

//#define CNN_USEOMP

#define CNN_USE_EXCEPTIONS

#ifdef CNN_USEOMP
#define CNN_TASK_SIZE 100
#else
#define CNN_TASK_SIZE 8
#endif

namespace cnn {
typedef double real_t
}
