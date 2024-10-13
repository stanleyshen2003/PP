#include <stdio.h>
#include <stdlib.h>
#include <thread>

typedef struct
{
    float x0, x1;
    float y0, y1;
    unsigned int width;
    unsigned int height;
    int maxIterations;
    int *output;
    int threadId;
    int numThreads;
    int totalPixels;
    int pixelsPerThread;
    float* xPos;
    float* yPos;

} WorkerArgs;

static inline int mandel(float c_re, float c_im, int count)
{
  float z_re = c_re, z_im = c_im;
  int i;
  for (i = 0; i < count; ++i)
  {

    if (z_re * z_re + z_im * z_im > 4.f)
        return i;

    float new_re = z_re * z_re - z_im * z_im;
    float new_im = 2.f * z_re * z_im;
    z_re = c_re + new_re;
    z_im = c_im + new_im;
  }

  return i;
}

void mandelbrotSerial(
    float x0, float y0, float x1, float y1,
    float* xPos, float* yPos,
    int width, int height,
    int startRow, int endRow,
    int startCol, int endCol,
    int maxIterations,
    int output[])
{

  int index = startRow * width + startCol;

  // first column
  for (int i = startCol; i < width; i++){
    float x = xPos[i];
    float y = yPos[startRow];
    output[index] = mandel(x, y, maxIterations);
    index++;
  }

  for (int j = startRow + 1; j < endRow; j++)
  {
    for (int i = 0; i < width; ++i)
    {
      float x = xPos[i];
      float y = yPos[j];

      output[index] = mandel(x, y, maxIterations);
      index++;
    }
  }

  for(int i = 0; i < endCol; i++){
    float x = xPos[i];
    float y = yPos[endRow];
    output[index] = mandel(x, y, maxIterations);
    index++;
  }
}

//
// workerThreadStart --
//
// Thread entrypoint.
void workerThreadStart(WorkerArgs *const args)
{

  // TODO FOR PP STUDENTS: Implement the body of the worker
  // thread here. Each thread could make a call to mandelbrotSerial()
  // to compute a part of the output image. For example, in a
  // program that uses two threads, thread 0 could compute the top
  // half of the image and thread 1 could compute the bottom half.
  // Of course, you can copy mandelbrotSerial() to this file and
  // modify it to pursue a better performance.

  int total_pixels = args->width * args->height;
  int pixels_per_thread = total_pixels / args->numThreads;
  int start_pixel = args->threadId * pixels_per_thread;
  int end_pixel = (args->threadId + 1) * pixels_per_thread;
  int startRow = start_pixel / args->width;
  int startCol = start_pixel % args->width;
  int endRow = end_pixel / args->width;
  int endCol = end_pixel % args->width;
  if (args->threadId == args->numThreads - 1){
      endRow = args->height;
      endCol = 0;
  }
  // printf("Thread %d: startRow: %d, endRow: %d, startCol: %d, endCol: %d\n", args->threadId, startRow, endRow, startCol, endCol);
  mandelbrotSerial(args->x0, args->y0, args->x1, args->y1, args->xPos, args->yPos, args->width, args->height, startRow, endRow, startCol, endCol, args->maxIterations, args->output);
}

//
// MandelbrotThread --
//
// Multi-threaded implementation of mandelbrot set image generation.
// Threads of execution are created by spawning std::threads.
void mandelbrotThread(
    int numThreads,
    float x0, float y0, float x1, float y1,
    int width, int height,
    int maxIterations, int output[])
{
    static constexpr int MAX_THREADS = 32;
    if (numThreads > MAX_THREADS)
    {
        fprintf(stderr, "Error: Max allowed threads is %d\n", MAX_THREADS);
        exit(1);
    }

    // Creates thread objects that do not yet represent a thread.
    std::thread workers[MAX_THREADS];
    WorkerArgs args[MAX_THREADS] = {};

    int total_pixels = width * height;
    int pixels_per_thread = total_pixels / numThreads;

    float dx = (x1 - x0) / width;
    float dy = (y1 - y0) / height;
    float *xPos = new float[width];
    float *yPos = new float[height];
    for (int i = 0; i < width; i++)
        xPos[i] = x0 + i * dx;
    for (int i = 0; i < height; i++)
        yPos[i] = y0 + i * dy;
    
    for (int i = 0; i < numThreads; i++)
    {
        // TODO FOR PP STUDENTS: You may or may not wish to modify
        // the per-thread arguments here.  The code below copies the
        // same arguments for each thread
        args[i].x0 = x0;
        args[i].y0 = y0;
        args[i].x1 = x1;
        args[i].y1 = y1;
        args[i].width = width;
        args[i].height = height;
        args[i].maxIterations = maxIterations;
        args[i].numThreads = numThreads;
        args[i].output = output;
        args[i].totalPixels = total_pixels;
        args[i].pixelsPerThread = pixels_per_thread;
        args[i].xPos = xPos;
        args[i].yPos = yPos;
        args[i].threadId = i;
    }

    // Spawn the worker threads.  Note that only numThreads-1 std::threads
    // are created and the main application thread is used as a worker
    // as well.
    for (int i = 1; i < numThreads; i++)
    {
        workers[i] = std::thread(workerThreadStart, &args[i]);
    }

    workerThreadStart(&args[0]);

    // join worker threads
    for (int i = 1; i < numThreads; i++)
    {
        workers[i].join();
    }
}