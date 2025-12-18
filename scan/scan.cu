#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host

  __shared__ float T[2*BLOCK_SIZE]; 

  // Load data into T before reduction 
  // Each thread will load 2 elements
  int idx = 2 * blockIdx.x * BLOCK_SIZE + threadIdx.x; 

  if (idx < len) {
    T[threadIdx.x] = input[idx]; 
  } else {
    T[threadIdx.x] = 0.0f;
  }

  if (idx + BLOCK_SIZE < len) {
    T[threadIdx.x + BLOCK_SIZE] = input[idx + BLOCK_SIZE]; 
  } else{
    T[threadIdx.x + BLOCK_SIZE] = 0.0f;
  }

  __syncthreads(); 

  // Step 1: Reduction step 

  int stride = 1; 
  
  while (stride < 2 * BLOCK_SIZE){
    __syncthreads(); 
    int index = (threadIdx.x + 1)*stride*2 - 1; 
    if (index < 2 * BLOCK_SIZE && (index - stride) >= 0){
      T[index] += T[index - stride]; 
    }

    stride = stride * 2; 
  }

  // Step 2: Reverse tree 
  stride = BLOCK_SIZE / 2; 
  
  while (stride > 0){
    __syncthreads(); 
    int index = (threadIdx.x+1)*stride*2 - 1; 
    if((index + stride) < 2*BLOCK_SIZE){
      T[index+stride] += T[index];  
    }
    stride = stride / 2; 
  } 

  __syncthreads(); 

  // Store results from shared memory to output 
  // Each thread writes back 2 elements 
  if (idx < len){
    output[idx] = T[threadIdx.x]; 
  } 

  if (idx + BLOCK_SIZE < len){
    output[idx + BLOCK_SIZE] = T[threadIdx.x + BLOCK_SIZE]; 
  }
}

__global__ void addBlockPrefixSums(float *output, float *blockPrefixSums, int len) {

  // blockPrefixSums is of size numBlocks -> # of blocks launched on the initial input for the scan kernel 
  // output is the deviceOutput after the first scan on the input

  // Each thread corrects the sum of two elements in output
  int idx = 2 * blockIdx.x * BLOCK_SIZE + threadIdx.x;
  
  // Block 0 doesn't need adjustment, as explained in lecture
  if (blockIdx.x > 0) {
    float blockSum = blockPrefixSums[blockIdx.x - 1];
    
    if (idx < len) {
      output[idx] += blockSum;
    }
    
    if (idx + BLOCK_SIZE < len) {
      output[idx + BLOCK_SIZE] += blockSum;
    }
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *blockSums;
  float *deviceBlockSums; 
  float *deviceBlockPrefixSums; 
  int numElements; // number of elements in the list
  int numBlocks;  

  args = wbArg_read(argc, argv);

  // Import data and create memory on host
  // The number of input elements in the input is numElements
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  numBlocks = (numElements + 2 * BLOCK_SIZE - 1)/(2 * BLOCK_SIZE); 


  // Allocate GPU memory.
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void**)&deviceBlockSums, numBlocks * sizeof(float)));
  wbCheck(cudaMalloc((void**)&deviceBlockPrefixSums, numBlocks * sizeof(float)));


  // Clear output memory.
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));

  // Copy input memory to the GPU.
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));

  //@@ Initialize the grid and block dimensions here
  dim3 block(BLOCK_SIZE); 
  dim3 grid(numBlocks); 


  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce

  // Kernel 1: Compute prefix sums for each block
  scan<<<grid,block>>>(deviceInput, deviceOutput, numElements); 
  cudaDeviceSynchronize();

  // Copy over the last sum for each block
  // Can either copy the entire deviceOutput array or only copy the last output sums 
  blockSums = (float *)malloc(numBlocks * sizeof(float));
  for (int i = 0; i < numBlocks; i++) {
    int lastIdx = min((i + 1) * 2 * BLOCK_SIZE, numElements) - 1;
    cudaMemcpy(&blockSums[i], &deviceOutput[lastIdx], sizeof(float), cudaMemcpyDeviceToHost);
  }

  // Kernel 2: As explained in the lecture, call the scan kernel again to compute prefix sum of the block sums
  // Can reuse the scan kernel here

  // Copy block sums to device and call scan kernel
  cudaMemcpy(deviceBlockSums, blockSums, numBlocks * sizeof(float), cudaMemcpyHostToDevice);
  dim3 blockSumGrid((numBlocks + 2*BLOCK_SIZE - 1)/(2*BLOCK_SIZE));
  scan<<<blockSumGrid, block>>>(deviceBlockSums, deviceBlockPrefixSums, numBlocks);  
  cudaDeviceSynchronize();

  // Kernel 3: Add block prefix sums and the correct the sums in the output of the first scan
  // Launch config for this kernel will be the same as the first scan 
  addBlockPrefixSums<<<grid,block>>>(deviceOutput, deviceBlockPrefixSums, numElements); 
  cudaDeviceSynchronize();

  // Copying output memory to the CPU
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));


  //@@  Free GPU Memory
  cudaFree(deviceInput); 
  cudaFree(deviceOutput); 
  cudaFree(deviceBlockSums);
  cudaFree(deviceBlockPrefixSums); 


  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);
  free(blockSums); 

  return 0;
}

