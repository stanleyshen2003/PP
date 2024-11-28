#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#define N 16

#include <cuda_runtime.h>

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int* d_img, int resX, int maxIterations) {
    int threadX = blockIdx.x * blockDim.x + threadIdx.x;
    int threadY = blockIdx.y * blockDim.y + threadIdx.y;


    float cX = lowerX + threadX * stepX;
    float cY = lowerY + threadY * stepY;

    float zX = cX, zY = cY;
    int i;
    for (i = 0; i < maxIterations; ++i) {
        float zX2 = zX * zX, zY2 = zY * zY;
        if (zX2 + zY2 > 4.0f) break;

        zY = 2.0f * zX * zY + cY;
        zX = zX2 - zY2 + cX;
    }

    d_img[threadY * resX + threadX] = i;
}


// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;
    int size = resX * resY * sizeof(int);

    int* ans;

    cudaHostRegister(img, size, cudaHostRegisterDefault);
    cudaHostGetDevicePointer(&ans, img, 0);

    // thread per block and block num
    dim3 threadsPerBlock(N, N);
    dim3 numBlocks(resX / threadsPerBlock.x, resY / threadsPerBlock.y);

    // launch kernel
    mandelKernel<<<numBlocks, threadsPerBlock>>>(lowerX, lowerY, stepX, stepY, ans, resX, maxIterations);

    cudaDeviceSynchronize();
    // cudaMemcpy(img, ans, size, cudaMemcpyDeviceToHost);
    cudaFree(ans);
    cudaHostUnregister(img);
}