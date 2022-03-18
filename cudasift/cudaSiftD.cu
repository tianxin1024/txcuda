#include "cudautils.h"
#include "cudaSiftD.h"
#include "cudaSift.h"


__constant__ int d_MaxNumPoints;
__device__ unsigned int d_PointCounter[8 * 2 + 1];
__constant__ float d_LowPassKernel[2 * LOWPASS_R + 1];
__constant__ float d_LaplaceKernel[8 * 12 * 16];


__global__ void LowPassBlock(float *d_Image, float *d_Result, int width, int pitch, int height) {
    __shared__ float xrows[16][32];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int xp = blockIdx.x * LOWPASS_W + tx;
    const int yp = blockIdx.y * LOWPASS_H + ty;
    const int N = 16;
    float *k = d_LowPassKernel;
    int xl = max(min(xp - 4, width - 1), 0);
#pragma unroll
    for (int l = -8; l < 4; l += 4) {
        int ly = l + ty;
        int yl = max(min(yp + l + 4, height - 1), 0);
        float val = d_Image[yl * pitch + xl];
        val = k[4] * ShiftDown(val, 4) + 
            k[3] * (ShiftDown(val, 5) + ShiftDown(val, 3)) +
            k[2] * (ShiftDown(val, 6) + ShiftDown(val, 2)) +
            k[1] * (ShiftDown(val, 7) + ShiftDown(val, 1)) +
            k[0] * (ShiftDown(val, 8) + val); 
        xrows[ly + 8][tx] = val;
    }
    __syncthreads();

#pragma unroll
    for (int l = 4; l < LOWPASS_H; l += 4) {
        int ly = l + ty;
        int yl = min(yp + l + 4, height - 1);
        float val = d_Image[yl * pitch + xl];
        val = k[4] * ShiftDown(val, 4) + 
            k[3] * (ShiftDown(val, 5) + ShiftDown(val, 3)) +
            k[2] * (ShiftDown(val, 6) + ShiftDown(val, 2)) +
            k[1] * (ShiftDown(val, 7) + ShiftDown(val, 1)) +
            k[0] * (ShiftDown(val, 8) + val); 
        xrows[(ly + 8) % N][tx] = val;
        int ys = yp + l - 4;
        if (xp < width && ys < height && tx < LOWPASS_W) {
            d_Result[ys * pitch + xp] = k[4] * xrows[(ly + 0) % N][tx] + 
                k[3] * (xrows[(ly - 1) % N][tx] + xrows[(ly + 1) % N][tx]) +
                k[2] * (xrows[(ly - 2) % N][tx] + xrows[(ly + 2) % N][tx]) +
                k[1] * (xrows[(ly - 3) % N][tx] + xrows[(ly + 3) % N][tx]) +
                k[0] * (xrows[(ly - 4) % N][tx] + xrows[(ly + 4) % N][tx]);
            __syncthreads();
        }

    }
        int ly = LOWPASS_H + ty;
        int ys = yp + LOWPASS_H - 4;
        if (xp < width && ys < height && tx < LOWPASS_W) {
            d_Result[ys * pitch + xp] = k[4] * xrows[(ly + 0) % N][tx] +
                k[3] * (xrows[(ly - 1) % N][tx] + xrows[(ly + 1) % N][tx]) +
                k[2] * (xrows[(ly - 2) % N][tx] + xrows[(ly + 2) % N][tx]) +
                k[1] * (xrows[(ly - 3) % N][tx] + xrows[(ly + 3) % N][tx]) +
                k[0] * (xrows[(ly - 4) % N][tx] + xrows[(ly + 4) % N][tx]);
        }

}
