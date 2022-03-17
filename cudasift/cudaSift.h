#ifndef CUDASIFT_H
#define CUDASIFT_H


#include "cudautils.h"
#include "cudaImage.h"


typedef struct SiftPoint {
    float xpos;
    float ypos;
    float scale;
    float sharpness;
    float edgeness;
    float orientation;
    float score;
    float ambiguity;
    int match;
    float match_xpos;
    float match_ypos;
    float match_error;
    float subsampling;
    float empty[3];
    float data[128];
} SiftPoint;


typedef struct SiftData {
    int numPts;         // Number of availables Sift points
    int maxPts;         // Number of allocated Sift points
#ifdef MANAGEDMEM
    SiftPoint *m_data;  // Managed data
#else
    SiftPoint *h_data;  // Host (CPU) data
    SiftPoint *d_data;  // Device (GPU) data
#endif
} SiftData;

void InitCuda(int devNum = 0);
void InitSiftData(SiftData &data,  int num = 1024, bool host = false, bool dev = true);

#endif  // end of CUDASIFT_H
