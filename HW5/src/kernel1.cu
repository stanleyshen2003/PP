#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#define N 16

// mandel function in serial.cpp
__device__ int mandel(float c_re, float c_im, int maxIteration) {
    
}



__global__ void mandelKernel (float lowerX, float lowerY, float stepX, float stepY, int* d_img, int resX, int resY, int maxIterations){
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    int threadX = blockIdx.x * blockDim.x + threadIdx.x;
    int threadY = blockIdx.y * blockDim.y + threadIdx.y;

    float x = lowerX + threadX * stepX;
    float y = lowerY + threadY * stepY;
    int index = threadY * resX + threadX;

    float z_re = c_re, z_im = c_im;
    int i;
    float new_re, new_im;
    for (i = 0; i < maxIteration; ++i) {
        new_re = z_re * z_re - z_im * z_im;
        if (new_re > 4.f)
            break;

        z_im = c_im + 2.f * z_re * z_im;
        z_re = c_re + new_re;
    }
    d_img[index] = i;
    
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int* ans;
    cudaMalloc((void**)&ans, resX * resY * sizeof(int));

    // thread per block and block num
    dim3 threadsPerBlock(N, N);
    dim3 numBlocks((resX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (resY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // launch kernel
    mandelKernel<<<numBlocks, threadsPerBlock>>>(lowerX, lowerY, stepX, stepY, ans, resX, resY, maxIterations);

    cudaMemcpy(img, ans, resX * resY * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(ans);
}