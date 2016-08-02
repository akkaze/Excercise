    #include <stdlib.h>  
    #include <stdio.h>  
    // 注意这里  
    #include <CL/cl.h>  
      
    #define LEN(arr) sizeof(arr) / sizeof(arr[0])  
    // 设备端kernel源程序，以字符串数组的方式保存，在某些论坛中提示每个语句最好以回车结束；  
    // 在运行时从源码编译成可执行在GPU上的kernel代码  
    const char* src[] = {   
        "__kernel void vec_add(__global const float *a, __global const float *b, __global float *c)\n",  
        "{\n",  
        "    int gid = get_global_id(0);\n",  
        "    c[gid] = a[gid] + b[gid];\n",  
        "}\n"  
    };  
      
    int main()  
    {  
        // 创建OpenCL context，该context与GPU设备关联，详见OpenCL规范4.3节  
        cl_context context = clCreateContextFromType(NULL, CL_DEVICE_TYPE_GPU, NULL, NULL, NULL);  
      
        // 获取context中的设备ID，详见OpenCL规范4.3节  
        size_t cb;  
        clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &cb);  
        cl_device_id *devices = (cl_device_id *)malloc(cb);  
        clGetContextInfo(context, CL_CONTEXT_DEVICES, cb, devices, NULL);  
      
        // create a command-queue，详见OpenCL规范5.1节  
        cl_command_queue cmd_queue = clCreateCommandQueue(context, devices[0], 0, NULL);  
      
        // 创建kernel，详见OpenCL规范5.4节和5.5.1节  
        cl_program program = clCreateProgramWithSource(context, LEN(src), src, NULL, NULL);  
        cl_int err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);  
        cl_kernel kernel = clCreateKernel(program, "vec_add", NULL);  
      
        // Host端输入初始化  
        size_t n = 5;  
        float srcA[] = {1, 2, 3, 4, 5};  
        float srcB[] = {5, 4, 3, 2, 1};  
        float dst[n];  
      
        // 设置kernel的输入输出参数，详见OpenCL规范5.2.1节  
        cl_mem memobjs[3];  
        memobjs[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, srcA, NULL);  
        memobjs[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, srcB, NULL);  
        memobjs[2] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * n, NULL, NULL);  
        // set "a", "b", "c" vector argument，详见OpenCL规范5.5.2节  
        err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memobjs[0]);  
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&memobjs[1]);  
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&memobjs[2]);  
      
        size_t global_work_size[1] = {n};  
        // execute kernel，详见OpenCL规范6.1节  
        err = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);  
      
        // read output array，详见OpenCL规范5.2.2节  
        err = clEnqueueReadBuffer(cmd_queue, memobjs[2], CL_TRUE, 0, n*sizeof(cl_float), dst, 0, NULL, NULL);  
      
        for (int i=0; i<n; ++i) {  
            printf("-> %.2f\n", dst[i]);  
        }  
        return 0;  
    }  
