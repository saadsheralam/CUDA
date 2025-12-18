#include <wb.h>

#define BLOCK_SIZE 512 //@@ This value is not fixed and you can adjust it according to the situation

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)
  
__global__ void total(float *input, float *output, int len) {
  //@@ Load a segment of the input vector into shared memory
  //@@ Traverse the reduction tree
  //@@ Write the computed sum of the block to the output vector at the correct index
  unsigned int idx = 2*blockDim.x*blockIdx.x + threadIdx.x;
  

  // Load data to shared memory 
  __shared__ float input_s[BLOCK_SIZE]; 

  // For last block idx or idx + blockDim.x can be out of bounds 
  float local = 0.0f;

  if (idx < len){
    local += input[idx];
  }

  if ((idx + blockDim.x) < len){
    local += input[idx + blockDim.x];
  } 

  input_s[threadIdx.x] = local;
  __syncthreads(); 

  // Reduction tree in shared memory 
  for (unsigned int stride = blockDim.x/2; stride > 0; stride/=2){
    if (threadIdx.x < stride){
      input_s[threadIdx.x] += input_s[threadIdx.x + stride]; 
    }
    __syncthreads(); 
  }

  // Final partial sum is at index 0 
  // Write partial sum to output using thread 0 
  if (threadIdx.x == 0){
    output[blockIdx.x] = input_s[0]; 
  }

}

int main(int argc, char **argv) {
  int ii;
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  //@@ Initialize device input and output pointers
  float *deviceInput; 
  float *deviceOutput; 

  int numInputElements;  // number of elements in the input list
  int numOutputElements; // number of elements in the output list

  args = wbArg_read(argc, argv);

  //Import data and create memory on host
  hostInput =
      (float *)wbImport(wbArg_getInputFile(args, 0), &numInputElements);

  numOutputElements = numInputElements / (BLOCK_SIZE << 1);
  if (numInputElements % (BLOCK_SIZE << 1)) {
    numOutputElements++;
  }
  hostOutput = (float *)malloc(numOutputElements * sizeof(float));

  // The number of input elements in the input is numInputElements
  // The number of output elements in the input is numOutputElements

  //@@ Allocate GPU memory
  cudaMalloc((void**) &deviceInput, numInputElements * sizeof(float)); 
  cudaMalloc((void**) &deviceOutput, numOutputElements * sizeof(float)); 


  //@@ Copy input memory to the GPU
  cudaMemcpy(deviceInput, hostInput, numInputElements * sizeof(float), cudaMemcpyHostToDevice);


  //@@ Initialize the grid and block dimensions here
  dim3 block(BLOCK_SIZE); 
  dim3 grid((numInputElements + (BLOCK_SIZE * 2) - 1)/(BLOCK_SIZE * 2)); 


  //@@ Launch the GPU Kernel and perform CUDA computation
  total<<<grid,block>>>(deviceInput, deviceOutput, numInputElements); 

  
  cudaDeviceSynchronize();  
  //@@ Copy the GPU output memory back to the CPU
  cudaMemcpy(hostOutput, deviceOutput, numOutputElements * sizeof(float), cudaMemcpyDeviceToHost); 

  
  /********************************************************************
   * Reduce output vector on the host
   * NOTE: One could also perform the reduction of the output vector
   * recursively and support any size input. 
   * For simplicity, we do not require that for this lab.
   ********************************************************************/
  for (ii = 1; ii < numOutputElements; ii++) {
    hostOutput[0] += hostOutput[ii];
  }

  //@@ Free the GPU memory
  cudaFree(deviceInput); 
  cudaFree(deviceOutput); 



  wbSolution(args, hostOutput, 1);

  free(hostInput);
  free(hostOutput);

  return 0;
}

