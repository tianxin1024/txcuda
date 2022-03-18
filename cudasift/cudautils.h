#ifndef CUDAUTILS_H
#define CUDAUTILS_H

#include <cstdio>
#include <cuda_runtime_api.h>
#include <iostream>
#include <cuda_runtime.h>

#define safeCall(err)  __safeCall(err, __FILE__, __LINE__)

inline void __safeCall(cudaError err, const char *file, const int line) {
    if (cudaSuccess != err) {
        fprintf(stderr, "safeCall() Runtime API error in file <%s>, line %i : %s.\n", file, line, cudaGetErrorString(err));
        exit(-1);
    }
}


inline bool deviceInit(int dev) {
    int deviceCount;
    safeCall(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
        return false;
    }
    if (dev < 0) dev = 0;
    if (dev > deviceCount - 1) {
        dev = deviceCount - 1;
    }
    cudaDeviceProp deviceProp;
    safeCall(cudaGetDeviceProperties(&deviceProp, dev));
    if (deviceProp.major < 1) {
        fprintf(stderr, "error: device does not support CUDA.\n");
        return false;
    }
    safeCall(cudaSetDevice(dev));
    return true;
}

class TimerGPU {
public:
    cudaEvent_t start, stop;
    cudaStream_t stream;
    TimerGPU(cudaStream_t stream_ = 0) : stream(stream_) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, stream);
    }
    ~TimerGPU() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    float read() {
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float time;
        cudaEventElapsedTime(&time, start, stop);
        return time;
    }
};


template <class T>
__device__ __inline__ T ShiftDown(T var, unsigned int delta, int width = 32) {
#if (CUDART_VERSION >= 9000)
    return __shfl_down_sync(0xffffffff, var, delta, width);
#else
    return __shfl_down(var, delta, width);
#endif
}



#endif  // end of CUDAUTILS_H
