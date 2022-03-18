#include <iostream>
#include "cudaSift.h"
#include "cudautils.h"
#include "cudaSiftD.h"
#include "cudaSiftH.h"

#include "cudaSiftD.cu"


void InitCuda(int devNum) {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    if (!nDevices) {
        std::cerr << "No CUDA devices available" << std::endl;
        return ;
    }

    devNum = std::min(nDevices - 1, devNum);
    deviceInit(devNum);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, devNum);
    std::cout << "\t Device Number: " << devNum << std::endl;
    std::cout << "\t Memory name: " << prop.name << std::endl;
    std::cout << "\t Memory Bus Width (bits): " << prop.memoryBusWidth << std::endl;
    printf("\t Peak Memory Bandwidth (GB/s): %.1f\n\n",
	            2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);

    return ;
}

float *AllocSiftTempMemory(int width, int height, int numOctaves, bool scaleUp) {
    TimerGPU timer(0);
    const int nd = NUM_SCALES + 3;
    int w = width * (scaleUp ? 2 : 1);
    int h = height * (scaleUp ? 2 : 1);
    int p = iAlignUp(w, 128);
    int size = h * p;           // image sizes
    int sizeTmp = nd * h * p;   // laplace buffer sizes
    for (int i = 0; i < numOctaves; i++) {
        w /= 2;
        h /= 2;
        int p = iAlignUp(w, 128);
        size += h * p;
        sizeTmp += nd * h * p;
    }
    float *memoryTmp = nullptr;
    size_t pitch;
    size += sizeTmp;
    safeCall(cudaMallocPitch((void **)&memoryTmp, &pitch, (size_t)4096, (size + 4095) / 4096 * sizeof(float)));
#ifndef VERBOSE
    printf("Allocated memory size: %d bytes\n", size);
    printf("Memory allocated time = \t %.2f ms \n\n", timer.read());
#endif
    return memoryTmp;
}

/* Multi-scale functions */
void PrepareLaplaceKernels(int numOctaves, float initBlur, float *kernel) {
    if (numOctaves > 1) {
        float totInitBlur = (float)sqrt(initBlur * initBlur + 0.5f * 0.5f) / 2.0f;
        PrepareLaplaceKernels(numOctaves - 1, totInitBlur, kernel);
    }
    float scale = pow(2.0f, -1.0f / NUM_SCALES);
    float diffScale = pow(2.0f, 1.0f / NUM_SCALES);
    for (int i = 0; i < NUM_SCALES + 3; i++) {
        float kernelSum = 0.0f;
        float var = scale * scale - initBlur * initBlur;
        for (int j = 0; j <= LAPLACE_R; j++) {
            kernel[numOctaves * 12 * 16 + 16 * i + j] = (float)expf(-(double)j *j / 2.0 / var);
            kernelSum += (j == 0 ? 1 : 2) * kernel[numOctaves * 12 * 16 + 16 * i + j];
        }
        for (int j = 0; j < LAPLACE_R; j++) {
            kernel[numOctaves * 12 * 16 + 16 * i + j] /= kernelSum;
        }
        scale *= diffScale;
    }
}

void ExtractSift(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, 
        float thresh, float lowestScale, bool scaleUp, float *tempMemory) {
    TimerGPU timer(0);
    unsigned int *d_PointCounterAddr;
    safeCall(cudaGetSymbolAddress((void **)&d_PointCounterAddr, d_PointCounter));
    safeCall(cudaMemset(d_PointCounterAddr, 0, (8 * 2 + 1) * sizeof(int)));
    safeCall(cudaMemcpyToSymbol(d_MaxNumPoints, &siftData.maxPts, sizeof(int)));

    const int nd = NUM_SCALES + 3;
    int w = img.width * (scaleUp ? 2 : 1);
    int h = img.height * (scaleUp ? 2 : 1);
    int p = iAlignUp(w, 128);
    int width = w, height = h;
    int size = h * p;
    int sizeTmp = nd * h * p;
    for (int i = 0; i < numOctaves; i++) {
        w /= 2;
        h /= 2;
        int p = iAlignUp(w, 128);
        size += h * p;
        sizeTmp += nd * h * p;
    }
    float *memoryTmp = tempMemory;
    size += sizeTmp;
    if (!tempMemory) {
        size_t pitch;
        safeCall(cudaMallocPitch((void **)&memoryTmp, &pitch, (size_t)4096, (size + 4095)/4096 * sizeof(float)));
#ifdef VERBOSE
    printf("Allocated memory size: %d bytes\n", size);
    printf("Memory allocated time = \t %.2f ms \n\n", timer.read());
#endif
    }

    float *memorySub = memoryTmp + sizeTmp;

    CudaImage lowImg;
    lowImg.Allocate(width, height, iAlignUp(width, 128), false, memorySub);
    if (!scaleUp) {
        float kernel[8 * 12 * 16];
        PrepareLaplaceKernels(numOctaves, 0.0f, kernel);
        safeCall(cudaMemcpyToSymbolAsync(d_LaplaceKernel, kernel, 8 * 12 * 16 * sizeof(float)));
        LowPass(lowImg, img, max(initBlur, 0.001f));
        TimerGPU timer1(0);
        ExtractSiftLoop(siftData, lowImg, numOctaves, 0.0f, thresh, lowestScale, 1.0f, memoryTmp, memorySub + height * iAlignUp(width, 128));
        safeCall(cudaMemcpy(&siftData.numPts, &d_PointCounterAddr[2 * numOctaves], sizeof(int), cudaMemcpyDeviceToHost));
        siftData.numPts = (siftData.numPts<siftData.maxPts ? siftData.numPts : siftData.maxPts);
        printf("SIFT extraction time = \t %.2f ms %d\n", timer1.read(), siftData.numPts);
    } else {
        CudaImage upImg;

    }

    if (!tempMemory) {
        safeCall(cudaFree(memoryTmp));
    }
#ifdef MANAGEDMEN
    safeCall(cudaDeviceSynchronize());
#else
    if (siftData.h_data) {
        safeCall(cudaMemcpy(siftData.h_data, siftData.d_data, sizeof(SiftPoint) * siftData.numPts, cudaMemcpyDeviceToHost));
    }
#endif
    double totTime = timer.read();
    printf("Incl prefiltering & memcpy = \t %.2f ms %d\n\n", totTime, siftData.numPts);
}

void InitSiftData(SiftData &data, int num, bool host, bool dev) {
    data.numPts = 0;
    data.maxPts = num;
    int sz = sizeof(SiftPoint) * num;
#ifdef MANAGEDMEN
    safeCall(cudaMallocManaged((void **) &data.m_data, sz));
#else
    data.h_data = nullptr;
    if (host) {
        data.h_data = (SiftPoint *)malloc(sz);
    }
    data.d_data = nullptr;
    if (dev) {
        safeCall(cudaMalloc((void **)&data.d_data, sz));
    }
#endif
}

double LowPass(CudaImage &res, CudaImage &src, float scale) {

    float kernel[2 * LOWPASS_R + 1];
    static float oldScale = -1.0f;
    if (scale != oldScale) {
        float kernelSum = 0.0f;
        float ivar2 = 1.0f / (2.0f * scale * scale);
        for (int j = -LOWPASS_R; j <= LOWPASS_R; j++) {
            kernel[j + LOWPASS_R] = (float)expf(-(double)j * j * ivar2);
            kernelSum += kernel[j + LOWPASS_R];
        }
        for (int j = -LOWPASS_R; j <= LOWPASS_R; j++) {
            kernel[j + LOWPASS_R] /= kernelSum;
        }
        safeCall(cudaMemcpyToSymbol(d_LowPassKernel, kernel, (2 * LOWPASS_R + 1) * sizeof(float)));
        oldScale = scale;
    }

    int width = res.width;
    int pitch = res.pitch;
    int height = res.height;
    dim3 blocks(iDivUp(width, LOWPASS_W), iDivUp(height, LOWPASS_H));
#if 1
    dim3 threads(LOWPASS_W + 2 * LOWPASS_R, 4);
    LowPassBlock<<<blocks, threads>>>(src.d_data, res.d_data, width, pitch, height);
#else
    LowPass<<<blocks, threads>>>(src.d_data, res.d_data, width, pitch, height);
#endif
    checkMsg("LowPass() execution failed\n");
    return 0.0;
}


int ExtractSiftLoop(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh,
        float lowestScale, float subsampling, float *memoryTmp, float *memorySub) {
#ifdef VERBOSE
    TimerGPU timer(0);
#endif
    int w = img.width;
    int h = img.height;
    if (numOctaves > 1) {
        CudaImage subImg;
        int p = iAlignUp(w / 2, 128);
        subImg.Allocate(w / 2, h / 2, p, false, memorySub);
        ScaleDown(subImg, img, 0.5f);
        float totInitBlur = (float)sqrt(initBlur * initBlur + 0.5f * 0.5f) / 2.0f;
        ExtractSiftLoop(siftData, subImg, numOctaves - 1, totInitBlur, thresh, lowestScale, subsmapling * 2.0f, memoryTmp, memorySub + (h / 2) * p);
    }
    ExtractSiftOctave(siftData, img, numOctaves, thresh, lowestScale, subsampling, memoryTmp);
#ifdef VERBOSE
    double totTime = timer.read();
    printf("ExtractSift time total = \t %.2f ms %d\n\n", totTime, numOctaves);
#endif
    return 0;
}

void ExtractSiftOctave(SiftData &siftData, CudaImage &img, int octave, float thresh, 
        float lowestScale, float subsampling, float *memoryTmp) {
    const int nd = NUM_SCALES + 3;
#ifdef VERBOSE
    unsigned int *d_PointCounterAddr;
    safeCall(cudaGetSymbolAddress((void**)&d_PointCounterAddr, d_PointCounter));
    unsigned int fstPts, totPts;
    safeCall(cudaMemcpy(&fstPts, &d_PointCounterAddr[2 * octave-1], sizeof(int), cudaMemcpyDeviceToHost)); 
    TimerGPU timer0;
#endif
    CudaImage diffImg[nd];
    int w = img.width; 
    int h = img.height;
    int p = iAlignUp(w, 128);
    for (int i=0;i<nd-1;i++) {
        diffImg[i].Allocate(w, h, p, false, memoryTmp + i*p*h); 
    }

    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = img.d_data;
    resDesc.res.pitch2D.width = img.width;
    resDesc.res.pitch2D.height = img.height;
    resDesc.res.pitch2D.pitchInBytes = img.pitch * sizeof(float);  
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0]   = cudaAddressModeClamp;
    texDesc.addressMode[1]   = cudaAddressModeClamp;
    texDesc.filterMode       = cudaFilterModeLinear;
    texDesc.readMode         = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;
    // Create texture object
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

#ifdef VERBOSE
    TimerGPU timer1;
#endif
    float baseBlur = pow(2.0f, -1.0f / NUM_SCALES);
    float diffScale = pow(2.0f, 1.0f / NUM_SCALES);
    LaplaceMulti(texObj, img, diffImg, octave); 
    FindPointsMulti(diffImg, siftData, thresh, 10.0f, 1.0f / NUM_SCALES, lowestScale / subsampling, subsampling, octave);
#ifdef VERBOSE
    double gpuTimeDoG = timer1.read();
    TimerGPU timer4;
#endif
    ComputeOrientations(texObj, img, siftData, octave); 
    ExtractSiftDescriptors(texObj, siftData, subsampling, octave); 
    //OrientAndExtract(texObj, siftData, subsampling, octave); 
  
    safeCall(cudaDestroyTextureObject(texObj));
#ifdef VERBOSE
    double gpuTimeSift = timer4.read();
    double totTime = timer0.read();
    printf("GPU time : %.2f ms + %.2f ms + %.2f ms = %.2f ms\n", totTime-gpuTimeDoG-gpuTimeSift, gpuTimeDoG, gpuTimeSift, totTime);
    safeCall(cudaMemcpy(&totPts, &d_PointCounterAddr[2*octave+1], sizeof(int), cudaMemcpyDeviceToHost));
    totPts = (totPts<siftData.maxPts ? totPts : siftData.maxPts);
    if (totPts>0) 
        printf("           %.2f ms / DoG,  %.4f ms / Sift,  #Sift = %d\n", gpuTimeDoG/NUM_SCALES, gpuTimeSift/(totPts-fstPts), totPts-fstPts); 
#endif
}

