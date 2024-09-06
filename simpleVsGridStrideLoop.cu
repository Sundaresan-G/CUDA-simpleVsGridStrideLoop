#include <stdio.h>
#include <assert.h>

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

template <typename T>
__global__ void simpleKernel(T* a)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  a[i] = a[i] + 1;
}

template <typename T>
__global__ void gridStrideLoopKernel(T* a, const int n)
{
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i+=gridDim.x*blockDim.x)
    a[i] = a[i] + 1;
}

template <typename T>
void runTest(int deviceId, int nMB, const cudaDeviceProp& prop)
{
  int blockSize = 512;
  float ms;

  T *d_a;
  cudaEvent_t startEvent, stopEvent;
    
  int n = nMB*1024*1024/sizeof(T);

  int numSM = prop.multiProcessorCount;
  int maxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;

  // NB:  d_a(33*nMB) for stride case
  checkCuda( cudaMalloc(&d_a, n * sizeof(T)) );

  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );

  printf("Simple Kernel results\n");

  printf("Block Size, Grid Size, Bandwidth (GB/s):\n");
  
  simpleKernel<<<(n + blockSize - 1)/blockSize, blockSize>>>(d_a); // warm up

  checkCuda(cudaDeviceSynchronize());

  for (int blockSize = 64; blockSize <= 1024; blockSize = blockSize << 1) {
    int gridSize = (n+blockSize -1)/blockSize;
    checkCuda( cudaMemset(d_a, 0, n * sizeof(T)) );

    checkCuda( cudaEventRecord(startEvent,0) );
    simpleKernel<<<gridSize, blockSize>>>(d_a);
    checkCuda( cudaEventRecord(stopEvent,0) );
    checkCuda( cudaEventSynchronize(stopEvent) );

    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
    printf("%10d, %9d, %f\n", blockSize, gridSize, 2*nMB/ms);
  }

  printf("\n");

  printf("Grid Stride Loop Kernel results\n");

  printf("Block Size, Grid Size, Bandwidth (GB/s):\n");
  
  gridStrideLoopKernel<<<n/blockSize, blockSize>>>(d_a, n); // warm up
  
  checkCuda(cudaDeviceSynchronize());

  for (int blockSize = 64; blockSize <= 1024; blockSize = blockSize << 1) {
    int gridSize = std::min(numSM * maxThreadsPerMultiProcessor/blockSize, (n+blockSize -1)/blockSize);
    checkCuda( cudaMemset(d_a, 0, n * sizeof(T)) );

    checkCuda( cudaEventRecord(startEvent,0) );
    gridStrideLoopKernel<<<gridSize, blockSize>>>(d_a, n);
    checkCuda( cudaEventRecord(stopEvent,0) );
    checkCuda( cudaEventSynchronize(stopEvent) );

    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
    printf("%10d, %9d, %f\n", blockSize, gridSize, 2*nMB/ms);
  }

  printf("\n");

  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );
  cudaFree(d_a);
}

int main(int argc, char **argv)
{
  int nMB = 12 * 10;
  int deviceId = 0;
  bool bFp64 = false;

  for (int i = 1; i < argc; i++) {    
    if (!strncmp(argv[i], "dev=", 4))
      deviceId = atoi((char*)(&argv[i][4]));
    else if (!strcmp(argv[i], "fp64"))
      bFp64 = true;
  }
  
  cudaDeviceProp prop;
  
  checkCuda( cudaSetDevice(deviceId) );

  checkCuda( cudaGetDeviceProperties(&prop, deviceId) );
  printf("Device: %s\n", prop.name);
  printf("maxThreadsPerMultiProcessor: %d\n", prop.maxThreadsPerMultiProcessor);
  printf("multiProcessorCount: %d\n", prop.multiProcessorCount);
  
  
  printf("Transfer size (MB): %d\n", nMB);
  
  printf("%s Precision\n", bFp64 ? "Double" : "Single");
  
  if (bFp64) runTest<double>(deviceId, nMB, prop);
  else       runTest<float>(deviceId, nMB, prop);
}