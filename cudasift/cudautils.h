#ifndef CUDAUTILS_H
#define CUDAUTILS_H


inline bool deviceInit(int dev) {
    int deviceCount;
    safeCall(cudaGetDeviceCount(&deviceCount));
}

#endif
