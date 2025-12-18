// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define THREADS_PER_BLOCK 256

//@@ insert code here

__global__ void cast_to_unsigned(float *input, unsigned char *output, int imageSize){

  int idx = blockIdx.x * blockDim.x + threadIdx.x; 
  
  // Some threads can be out of bound in the last block 
  if (idx < imageSize) {
    output[idx] = (unsigned char) (255 * input[idx]);  
  }
}

__global__ void convert_to_grayscale(unsigned char *input, unsigned char *output, int imageSize, int numChannels){

  int idx = blockIdx.x * blockDim.x + threadIdx.x; 

  if (idx < imageSize) {

    // Extract RGB channels 
    unsigned char r = input[idx * numChannels]; 
    unsigned char g = input[idx * numChannels + 1]; 
    unsigned char b = input[idx * numChannels + 2]; 
    
    output[idx] = (unsigned char) (0.21*r + 0.71*g + 0.07*b); 
  }
}

__global__ void compute_histogram(unsigned char *input, unsigned int* output, int imageSize) {

  // Each block computes its own private histogram 
  __shared__ unsigned int histo_private[HISTOGRAM_LENGTH]; 

  // Block is launched with 256 threads 
  // Initialize private histo to 0
  if (threadIdx.x < HISTOGRAM_LENGTH){
    histo_private[threadIdx.x] = 0; 
  }

  __syncthreads(); 

  int idx = blockIdx.x * blockDim.x + threadIdx.x; 

  // Threads load consecutive elements from input for coalesced memory accesses 
  int stride = blockDim.x * gridDim.x; 
  while (idx < imageSize){
    atomicAdd(&histo_private[input[idx]], 1); 
    idx += stride; 
  }

  __syncthreads(); 

  // Copy data from private histo to output histogram
  if (threadIdx.x < HISTOGRAM_LENGTH){
    atomicAdd(&output[threadIdx.x], histo_private[threadIdx.x]); 
  }
}

__device__ float clamp(float x, float start, float end){
  return fminf(fmaxf(x, start), end); 
} 

__device__ float correct_color(float* cdf, float cdf_min, unsigned int val){
  return clamp(255 * (cdf[val] - cdf_min)/(1.0 - cdf_min), 0.0f, 255.0f); 
}

__global__ void apply_equalization(unsigned char *input, float *output, int imageSize, float *cdf, float cdf_min){

  int idx = blockIdx.x * blockDim.x + threadIdx.x; 

  // I am also casting to float in the same step as equalization
  if(idx < imageSize){
    output[idx] = (float) (correct_color(cdf, cdf_min, input[idx])) / 255.0; 
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  int imageSize; 
  int grayImageSize; 
  float *deviceInputImageData; 
  unsigned char *deviceUnsignedCharOutputData;
  unsigned char *deviceGrayScaleOutputData; 
  unsigned int *deviceHistogramOutputData; 
  unsigned int *hostHistogramData; 
  float *hostCdf; 
  float *deviceCdf; 
  float cdf_min; 
  float *deviceEqualizationOutputData; 

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  //Import data and create memory on host
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);  


  //@@ insert code here

  imageSize = imageWidth * imageHeight * imageChannels;
  grayImageSize = imageWidth * imageHeight;

  // Allocate memory on host 
  hostOutputImageData = (float*)malloc(imageSize * sizeof(float));
  hostHistogramData = (unsigned int*)malloc(HISTOGRAM_LENGTH * sizeof(unsigned int)); 
  hostCdf = (float*)malloc(HISTOGRAM_LENGTH * sizeof(float)); 

  // Allocate memory on device
  cudaMalloc((void **) &deviceInputImageData, imageSize * sizeof(float)); 
  cudaMalloc((void **) &deviceUnsignedCharOutputData, imageSize * sizeof(unsigned char)); 
  cudaMalloc((void **) &deviceGrayScaleOutputData, grayImageSize * sizeof(unsigned char)); 
  cudaMalloc((void **) &deviceHistogramOutputData, HISTOGRAM_LENGTH * sizeof(unsigned int)); 
  cudaMalloc((void **) &deviceCdf, HISTOGRAM_LENGTH * sizeof(float)); 
  cudaMalloc((void **) &deviceEqualizationOutputData, imageSize * sizeof(float));

  // Copy data to device 
  cudaMemcpy(deviceInputImageData, hostInputImageData, imageSize * sizeof(float), cudaMemcpyHostToDevice); 

  // Step 1: Launch kernel to convert image to unsigned 
  dim3 block(THREADS_PER_BLOCK); 
  dim3 grid((imageSize + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK); 
  cast_to_unsigned<<<grid,block>>>(deviceInputImageData, deviceUnsignedCharOutputData, imageSize); 

  // Step 2: Launch kernel to convert image to gray scale
  dim3 grayGrid((grayImageSize + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK); 
  convert_to_grayscale<<<grayGrid,block>>>(deviceUnsignedCharOutputData, deviceGrayScaleOutputData, grayImageSize, imageChannels); 

  // Step 3: Launch kernel to calculate histogram for grayscale image 

  // Zero out the deviceHistogramOutputData, which is unitialized on the device 
  cudaMemset(deviceHistogramOutputData, 0, HISTOGRAM_LENGTH * sizeof(unsigned int));

  compute_histogram<<<grayGrid,block>>>(deviceGrayScaleOutputData, deviceHistogramOutputData, grayImageSize); 
  cudaMemcpy(hostHistogramData, deviceHistogramOutputData, HISTOGRAM_LENGTH * sizeof(unsigned int), cudaMemcpyDeviceToHost); 
  
  // Step 4: Compute CDF on Host 
  hostCdf[0] = (float)hostHistogramData[0] / grayImageSize; 
  for (int i = 1; i < HISTOGRAM_LENGTH; ++i){
    hostCdf[i] = hostCdf[i-1] + ((float)hostHistogramData[i] / grayImageSize); 
  }

  // Step 5: Compute CDF Min 
  cdf_min = 1.0f; 
  for (int i = 0; i < HISTOGRAM_LENGTH; ++i) {
    if (hostCdf[i] > 0 && hostCdf[i] < cdf_min) {
        cdf_min = hostCdf[i];
    }
  }

  // Step 6: Launch a kernel to apply equalization function to RGB image and cast to float
  cudaMemcpy(deviceCdf, hostCdf, HISTOGRAM_LENGTH * sizeof(float), cudaMemcpyHostToDevice); 
  apply_equalization<<<grid, block>>>(deviceUnsignedCharOutputData, deviceEqualizationOutputData, imageSize, deviceCdf, cdf_min); 
  cudaMemcpy(hostOutputImageData, deviceEqualizationOutputData, imageSize * sizeof(float), cudaMemcpyDeviceToHost); 

  wbImage_setData(outputImage, hostOutputImageData);
  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(deviceInputImageData); 
  cudaFree(deviceUnsignedCharOutputData); 
  cudaFree(deviceGrayScaleOutputData); 
  cudaFree(deviceHistogramOutputData); 
  cudaFree(deviceCdf) ; 
  cudaFree(deviceEqualizationOutputData); 
  free(hostOutputImageData); 
  free(hostHistogramData); 
  free(hostCdf); 

  return 0;
}

