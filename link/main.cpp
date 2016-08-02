#include "Test.h"
#include <stdlib.h>
#include <iostream>
#include <boost/timer/timer.hpp>

using namespace std;
int main(int argc, char *argv[]) {
        int n = 200000000;
        size_t size = n * sizeof(float);
        int i;

        float *h_a = (float*)malloc(size), *h_b = (float*)malloc(size), *h_c = (float*)malloc(size);

        for(i = 0; i < n; i++) {
                h_a[i] = h_b[i] = i;
        }
        boost::timer::cpu_timer timer;
				Test test;
				timer.start();
        test.addVec_gpu(h_a, h_b, h_c, n);
        cout<<timer.format()<<endl;
				timer.start();
        test.addVec_cpu(h_a, h_b, h_c, n);
        cout<<timer.format()<<endl;
//        for(int i=0;i<n;++i)
//				std::cout << h_c[i] << '\t';
				free(h_a);
				free(h_b);
				free(h_c);
        exit(0);
}
