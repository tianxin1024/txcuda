#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

struct ConstantValue {
    unsigned char color[3];
};


// 设计为2维度的核gridDim.x, blockDim.x
// gridDim -> dim3
// dim3 gridDim = 123
// gridDim.? = 123;  gridDim.x = 123
// 这个核函数需要启动多少个线程
// 图像大小个线程, 每次核函数只需要处理一个像素
// size.width  是图像的宽度
// size.height 使图像的高度
// gridDim = ceil(size.width * size.height / 512)
// blockDim = 512  //  最好是整除32， 一般如果是二维，这里可以取512/256
// 1. jobs, 线程数 = size.width * size.height
// 2. 执行warp_affine_gpu_impl 核函数的次数， 是不是就是线程数
//       -- 那么是不是存在超出图像边界的执行
//       -- 需要控制边界
//   使用的是输出图像的大小
__global__ void warp_affine_impl(
        unsigned char* src,  // 来源图像
        unsigned char* dst,  // 目标图像
        float* M,             // dst映射到src的矩阵
        int src_width, int src_height,
        int dst_width, int dst_height,
        int edge             // 边界
        ) {

    /*
       shape 
       gridDim.z = 1
       gridDim.y = 1
       girdDim.x = N
       blockDim.z = 1
       blockDim.y = 1
       blockDim.x = M

       position
       blockIdx.z = 0
       blockIdx.y = 0
       blockIdx.x = 1
       

    */
}


Mat warp_affine_gpu(const Mat& image, const Mat& M, const Size& size) {

    // 构建输出的Mat
    // Mat(rows, cols, int type);
    Mat output(size.height, size.width, image.type());

    // M的定义是2x3矩阵， ix -> dx, dx -> ix
    // M的定义是ix -> dx, 也就是图像变换到目标的矩阵
    //  dx = M[0] * ix + M[1] * iy + M[2]
    // 1. 处理M, 使其变为 dx -> ix 的映射
    // M.数据类型， double类型的， float64

    Mat iM;
    cv::invertAffineTransform(M, iM);

    // 2. 对iM做类型转换， 转为float
    iM.converTo(iM, CV::32F);

    // 3. 准备device 指针
    unsigned char* src_device = nullptr;
    unsigned char* dst_device = nullptr;
    float* iM_device = nullptr;


    // 4. 分配设备空间
    // image 数据类型， unsigned char -> uint8, byte = 1
    // image bytes = width * height * channels * element_byte
    // 这句话[iamge.step.p[0] * image.size[0]]  是万能写法
    //    -- 只要你想获取Mat中数据所占字节数，这句话满足你

    size_t src_bytes = image.step.p[0] * image.size[0];
    size_t dst_bytes = output.step.p[0] * output.size[0];
    size_t iM_bytes = iM.step.p[0] * iM.size[0];
    cudaMalloc(&src_device, src_bytes);
    cudaMalloc(&dst_device, dst_bytes);
    cudaMalloc(&iM_device, iM_bytes);

    // 5. 把数据从主机复制到设备
    // image.ptr<uchar>(0)   行号为0的地址
    // image.data            图像的起始地址
    /* cudaMemcpy(src_device, image.ptr<uchar>(0), src_bytes, cudaMemcpyHostToDevice); */
    cudaMemcpy(src_device, image.data, src_bytes, cudaMemcpyHostToDevice);


    // 6. 执行核函数
    int jobs = size.area();
    int threads = 512;
    int blocks = ceil(jobs / (float)threads);

    // gridDim, blockDim, memory,  stream
    warp_affine_gpu_impl<<<blocks, threads, 0, nullptr>>>(
            src_device, dst_device, iM_device, 
            image.cols, image.rows, 
            size.width, size.height, jobs);

    // 7. 计算结果复制回来
    cudaMemcpy(output.data, dst_device, dst_bytes, cudaMemcpyDeviceToHost);


    // 8. 清除分配的空间
    cudaFree(src_device);
    cudaFree(dst_device);
    cudaFree(iM_device);

    return output;
}

