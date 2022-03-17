#include "cudautils.h"
#include "cudaSiftD.h"
#include "cudaSift.h"


__constant__ int d_MaxNumPoints;
__device__ unsigned int d_PointCounter[8 * 2 + 1];
__constant__ float d_LaplaceKernel[8 * 12 * 16];
