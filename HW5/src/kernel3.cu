#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#define N 16

__global__ void mandelKernel (float lowerX, float lowerY, float stepX, float stepY, int* d_img, int resX, size_t pitch, int maxIterations){
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    int threadX = blockIdx.x * blockDim.x + threadIdx.x * 2;
    int threadY = blockIdx.y * blockDim.y + threadIdx.y;

    float x = lowerX + threadX * stepX;
    float y = lowerY + threadY * stepY;

    for (int j = 0; j < 2; j++) {
        x = x + j * stepX;
        float z_re = x, z_im = y;
        int i;
        float new_re;
        for (i = 0; i < maxIterations; ++i) {
            new_re = z_re * z_re - z_im * z_im;
            if (new_re > 4.f)
                break;

            z_im = y + 2.f * z_re * z_im;
            z_re = x + new_re;
        }
        *((int*)((char*)d_img + threadY * pitch) + threadX + j) = i;
    }

    
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int* host_image, *pinnedImg;
    int size = resX * resY * sizeof(int);
    size_t pitch;
    cudaHostAlloc((void**)&host_image, size, cudaHostAllocDefault);
    cudaMallocPitch((void**)&pinnedImg, &pitch, resX * sizeof(int), resY);


    // thread per block and block num
    dim3 threadsPerBlock(N / 2, N);
    dim3 numBlocks((resX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (resY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // launch kernel
    mandelKernel<<<numBlocks, threadsPerBlock>>>(lowerX, lowerY, stepX, stepY, pinnedImg, resX, pitch, maxIterations);

    cudaMemcpy2D(img, resX * sizeof(int), pinnedImg, pitch, resX * sizeof(int), resY, cudaMemcpyDeviceToHost);
    // memcpy(img, host_image, size);
    cudaFree(pinnedImg);
    cudaFreeHost(host_image);
}