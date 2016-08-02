#include "Test.h"
#include <math.h>
void Test::addVec_cpu(float *a, float *b, float *c, int n) {
              for(int i=0;i<n;++i)
              {
                c[i] = sqrt(a[i] * b[i]);
                }
}


