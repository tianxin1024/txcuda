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


#endif  // end of CUDASIFTH_H
