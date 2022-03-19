#include <cstdio>
#include "cudautils.h"
#include "cudaImage.h"

void CudaImage::Allocate(int w, 
                         int h, 
                         int p, 
                         bool host, 
                         float *devmem, 
                         float *hostmem) {
    width = w;
    height = h;
    pitch = p;
    d_data = devmem;
    h_data = hostmem;
    t_data = nullptr;

    if (devmem == nullptr) {
        safeCall(cudaMallocPitch((void **)&d_data, (size_t*)&pitch, (size_t)(sizeof(float) * width), (size_t)height));
        pitch /= sizeof(float);
        if (d_data == nullptr) {
            printf("Failed to allocate device data\n");
            d_internalAlloc = true;
        }

        if (host && hostmem == nullptr) {
            h_data = (float *)malloc(sizeof(float) * pitch * height);
            h_internalAlloc = true;
        }
    }
}

CudaImage::CudaImage() : width(0), height(0), d_data(nullptr), t_data(nullptr), 
                d_internalAlloc(false), h_internalAlloc(false) {}

CudaImage::~CudaImage() {
    if (d_internalAlloc && d_data != nullptr) {
        safeCall(cudaFree(d_data));
    }
    d_data = nullptr;
    if (h_internalAlloc && h_data != nullptr) {
        free(h_data);
    }
    h_data = nullptr;
    if (t_data != nullptr) {
        safeCall(cudaFreeArray((cudaArray *)t_data));
    }
    t_data = nullptr;
}


double CudaImage::Download() {
    TimerGPU timer(0);
    int p = sizeof(float) * pitch;
    if (d_data != nullptr && h_data != nullptr) {
        safeCall(cudaMemcpy2D(d_data, p, h_data, sizeof(float) * width, sizeof(float) * width, height, cudaMemcpyHostToDevice));
    }
    double gpuTime = timer.read();
#ifdef VERBOSE
    printf("Download  time = \t \t %.2f ms\n", gpuTime);
#endif
    return gpuTime;
}

double CudaImage::Readback() {
    TimerGPU timer(0);
    int p = sizeof(float) * pitch;
    safeCall(cudaMemcpy2D(h_data, sizeof(float) * width, d_data, p, sizeof(float) * width, height, cudaMemcpyDeviceToHost));
    double gpuTime = timer.read();
#ifndef VERBOSE
    printf("Readback time = \t\t %.2f ms\n", gpuTime);
#endif
    return gpuTime;
}

double CudaImage::InitTexture() {
    TimerGPU timer(0);
    cudaChannelFormatDesc t_desc = cudaCreateChannelDesc<float>();
    safeCall(cudaMallocArray((cudaArray **)&t_data, &t_desc, pitch, height));
    if (t_data == nullptr) {
        printf("Failed to allocated texture data\n");
    }
    double gpuTime = timer.read();
#ifdef VERBOSE
    printf("InitTexture time = \t\t  %.2f ms\n", gpuTime);
#endif
    return gpuTime;
}


double CudaImage::CopyToTexture(CudaImage &dst, bool host) {
    if (dst.t_data == nullptr) {
        printf("Error CopyToTexture: No texture data\n");
        return 0.0;
    }
    if ((!host || h_data == nullptr) && (host || d_data == nullptr)) {
        printf("Error CopyToTexture: No source data\n");
        return 0.0;
    }
    TimerGPU timer(0);
    if (host) {
        safeCall(cudaMemcpyToArray((cudaArray *)dst.t_data, 0, 0, h_data, 
                    sizeof(float) * pitch * dst.height, cudaMemcpyHostToDevice));
    } else {
        safeCall(cudaMemcpyToArray((cudaArray *)dst.t_data, 0, 0, d_data,
                sizeof(float) * pitch * dst.height, cudaMemcpyDeviceToHost));
    }
    safeCall(cudaDeviceSynchronize());
    double gpuTime = timer.read();
#ifdef VERBOSE
    printf("CopyToTexture  time = \t\t  %.2f ms\n", gpuTime);
#endif
    return gpuTime;
}


int iDivUp(int a, int b) {
    return (a%b != 0) ? (a/b + 1) : (a/b);
}

int iDivDown(int a, int b) {
    return a / b;
}

int iAlignUp(int a, int b) {
    return (a%b != 0) ? (a - a % b + b) : a;
}

int iAlignDown(int a, int b) {
    return a - a%b;
}

