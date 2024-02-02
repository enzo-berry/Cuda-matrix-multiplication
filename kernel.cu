#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// GPU Matrix multiplication
__global__ void matrixMulKernelGPU(int* c, const int* a, const int* b, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int sum = 0;
        for (int k = 0; k < n; k++) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

// CPU Matrix multiplication
void matrixMulCPU(int* c, const int* a, const int* b, int n) {
	for (int row = 0; row < n; row++) {
		for (int col = 0; col < n; col++) {
			int sum = 0;
			for (int k = 0; k < n; k++) {
				sum += a[row * n + k] * b[k * n + col];
			}
			c[row * n + col] = sum;
		}
	}
}

void randomInts(int* a, int size) {
    for (int i = 0; i < size; i++) {
        a[i] = rand() % 100; // Random numbers between 0 and 99
    }
}

void MatMultBench(int size) {
    int* a, * b, * c; // Host copies of a, b, c
    int* d_a, * d_b, * d_c; // Device copies of a, b, c

    int byte_size = size * size * sizeof(int);
    clock_t startCPU, endCPU;
    cudaEvent_t start, stop;
    float GPU_time, CPU_time;

    // Initialize CUDA events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Print
    printf("Matrix size: %d x %d\n", size, size);

    // Allocate space for device copies of a, b, c
    cudaMalloc((void**)&d_a, byte_size);
    cudaMalloc((void**)&d_b, byte_size);
    cudaMalloc((void**)&d_c, byte_size);

    // Allocate space for host copies of a, b, c and setup input values
    a = (int*)malloc(byte_size); randomInts(a, size * size);
    b = (int*)malloc(byte_size); randomInts(b, size * size);
    c = (int*)malloc(byte_size);

    // Start GPU timing
    cudaEventRecord(start);

    // Copy inputs to device
    cudaMemcpy(d_a, a, byte_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, byte_size, cudaMemcpyHostToDevice);

    // Setup the execution configuration
    dim3 threadsPerBlock(16, 16); // 16x16 is a common choice, modify as needed
    dim3 blocksPerGrid((size + threadsPerBlock.x - 1) / threadsPerBlock.x, (size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel on the GPU
    matrixMulKernelGPU << <blocksPerGrid, threadsPerBlock >> > (d_c, d_a, d_b, size);

    // Copy result back to host
    cudaMemcpy(c, d_c, byte_size, cudaMemcpyDeviceToHost);

    // Stop GPU timing
    cudaEventRecord(stop);

    // Wait for GPU to finish
    cudaEventSynchronize(stop);

    // Calculate GPU time
    cudaEventElapsedTime(&GPU_time, start, stop);
    printf("Time taken for GPU: %f ms\n", GPU_time);

    // CPU Matrix multiplication, monitor time with libc
    startCPU = clock();
    matrixMulCPU(c, a, b, size);
    endCPU = clock();
    CPU_time = (float)(endCPU - startCPU) * 1000 / CLOCKS_PER_SEC;
    printf("Time taken for CPU: %f ms\n\n", CPU_time);

    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(a); free(b); free(c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


int main() {
	// 1, 2, 4, 8, 16, ... until 4096
    for (int i = 1; i <= 4096; i *= 2) {
		MatMultBench(i);
	}
    
    return 0;
}
