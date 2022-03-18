#ifndef CUDASIFTH_H
#define CUDASIFTH_H

#include "cudautils.h"
#include "cudaImage.h"
#include "cudaSift.h"

int ExtractSiftLoop(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh,
        float lowestScale, float subsampling, float *memoryTmp, float *memorySub);

void ExtractSiftOctave(SiftData &siftData, CudaImage &img, int octave, float thresh, 
        float lowestScale, float subsampling, float *memoryTmp);

double LowPass(CudaImage &res, CudaImage &src, float scale);

double ScaleDown(CudaImage &res, CudaImage &src, float variance);

double LaplaceMulti(cudaTextureObject_t texObj, CudaImage &baseImage, 
        CudaImage * results, int octave);

double FindPointsMulti(CudaImage *sources, SiftData &siftData, float thresh, float edgeLimit,
        float factor, float lowestScale, float subsampling, int octave);

double ExtractSiftDescriptors(cudaTextureObject_t texObj, SiftData &siftData, float subsampling, int octave);

double ComputeOrientations(cudaTextureObject_t texObj, CudaImage &src, SiftData &siftData, int octave);

#endif  // end of CUDASIFTH_H
