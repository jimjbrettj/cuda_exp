 #include <stdio.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


/*
Common workflow of cuda programs:
    1) Allocate host memory and initialized host data
    2) Allocate device memory
    3) Transfer input data from host to device memory
    4) Execute kernels
    5) Transfer output from device memory to host
*/

__global__ void add(int a, int b, int *c){
    *c = a + b;
}

__global__ void test(int a, int *c){
    *c = a;
}

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

int main(){
    // Allocate host memory and initialized host data
    int a, b, c;
    int *dev_c;
    a = 3;
    b = 4;
    // Allocate device memory
    gpuErrchk(cudaMalloc((void**) &dev_c, sizeof(int)));
    // Execute kernels
    cuda_hello<<<1,1>>>();

    cudaDeviceSynchronize();
    //test<<<1,1>>>(a, dev_c);
    add<<<1,1>>>(a, b, dev_c);
    // Transfer output from device memory to host
    cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d + %d = %d\n", a, b, c);
    gpuErrchk( cudaPeekAtLastError() );
    // Free device memory
    cudaFree(dev_c);
    return 0;
}

// #define N 10

// __global__ void vector_add(float *out, float *a, float *b, int n) {
//     for(int i = 0; i < n; i++){
//         out[i] = a[i] + b[i];
//     }
// }

// int main(){
//     float *a, *b, *out; 

//     // Allocate memory
//     a   = (float*)malloc(sizeof(float) * N);
//     b   = (float*)malloc(sizeof(float) * N);
//     out = (float*)malloc(sizeof(float) * N);

//     // Initialize array
//     for(int i = 0; i < N; i++){
//         a[i] = 1.0f; b[i] = 2.0f;
//     }

//     // Main function
//     //vector_add(out, a, b, N);
//     vector_add<<<1,1>>>(out, a, b, N);

//     for(int i = 0; i < N; i++){
//         printf("%f ", out[i]);
//     }
//     printf("\n");
//    	return 0;
// }
