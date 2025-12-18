#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define KERNEL_WIDTH 3
#define KERNEL_SIZE 27
#define TILE_WIDTH 8

//@@ Define constant memory for device kernel here
__constant__ float kernel_c[KERNEL_WIDTH][KERNEL_WIDTH][KERNEL_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
              
  const int KW = KERNEL_WIDTH; 
  const int radius = KW / 2;

  // Local thread indices (0 .. TILE_WIDTH+1)
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  int gx = blockIdx.x * TILE_WIDTH + tx - radius;
  int gy = blockIdx.y * TILE_WIDTH + ty - radius;
  int gz = blockIdx.z * TILE_WIDTH + tz - radius;

  // Allocate shared memory for tile
  // This will take 1000 * 4 = 4KB 
  __shared__ float tile[TILE_WIDTH + 2][TILE_WIDTH + 2][TILE_WIDTH + 2];

  // Load tile into shared memory
  if (gx >= 0 && gx < x_size && gy >= 0 && gy < y_size && gz >= 0 && gz < z_size) {
    tile[tx][ty][tz] = input[gz * (y_size * x_size) + gy * x_size + gx];
  } else {
    tile[tx][ty][tz] = 0.0f;
  }

  __syncthreads();

  // Only threads within the input size will perform computation
  if (tx >= radius && tx <= radius + TILE_WIDTH - 1 &&
      ty >= radius && ty <= radius + TILE_WIDTH - 1 &&
      tz >= radius && tz <= radius + TILE_WIDTH - 1) {

      float P_value = 0.0f;

      for (int kz = 0; kz < KW; ++kz) {
        for (int ky = 0; ky < KW; ++ky) {
          for (int kx = 0; kx < KW; ++kx) {
            int sx = tx + kx - radius;
            int sy = ty + ky - radius;
            int sz = tz + kz - radius;
            P_value += kernel_c[kz][ky][kx] * tile[sx][sy][sz];
          }
        }
      }

      if (gx >= 0 && gx < x_size &&
          gy >= 0 && gy < y_size &&
          gz >= 0 && gz < z_size) {
            output[gz * (y_size * x_size) + gy * x_size + gx] = P_value;
      }
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  //@@ Initial deviceInput and deviceOutput here.
  float *deviceInput; 
  float *deviceOutput; 

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  int output_size = z_size * y_size * x_size; 

  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc((void **) &deviceInput, output_size * sizeof(float)); 
  cudaMalloc((void **) &deviceOutput, output_size * sizeof(float)); 



  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions
  cudaMemcpy(deviceInput, hostInput + 3, output_size * sizeof(float), cudaMemcpyHostToDevice); 
  cudaMemcpyToSymbol(kernel_c, hostKernel, KERNEL_SIZE * sizeof(float)); 



  //@@ Initialize grid and block dimensions here
  // I am implementing strategy 2 from the lecture. 
  // Parallelizing on tile loading rather than the output. 
  // Since the kernel is fixed (3x3x3), I've hardcoded + 2 (2 * kernel radius)
  // There will be extra threads to fetch halos, but these threads won't participate in computation
  dim3 block(TILE_WIDTH + 2, TILE_WIDTH + 2, TILE_WIDTH + 2); 
  dim3 grid(
    (x_size + TILE_WIDTH - 1) / TILE_WIDTH, 
    (y_size + TILE_WIDTH - 1) / TILE_WIDTH, 
    (z_size + TILE_WIDTH - 1) / TILE_WIDTH
  );

  //@@ Launch the GPU kernel here
  conv3d<<<grid, block>>>(deviceInput, deviceOutput, z_size, y_size, x_size);

  cudaDeviceSynchronize();



  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostOutput + 3, deviceOutput, output_size * sizeof(float), cudaMemcpyDeviceToHost); 




  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  //@@ Free device memory
  cudaFree(deviceInput); 
  cudaFree(deviceOutput);


  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}

