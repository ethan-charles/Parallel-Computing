#include <iostream>
#include <cmath>
#include <cuda.h>


__global__ void matrix_update(double **A, double **Ao, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i > 0 && i < n - 1 && j > 0 && j < n - 1) {
        double neighbors[4] = {Ao[i + 1][j + 1], Ao[i + 1][j - 1], Ao[i - 1][j + 1], Ao[i - 1][j - 1]};
        double min1 = neighbors[0], min2 = neighbors[1];
        if (min2 < min1) {
            double temp = min1;
            min1 = min2;
            min2 = temp;
        }
        for (int k = 2; k < 4; k++) {
            if (neighbors[k] < min1) {
                min2 = min1;
                min1 = neighbors[k];
            } else if (neighbors[k] < min2) {
                min2 = neighbors[k];
            }
        }
        A[i][j] = Ao[i][j] + min2;
    }
}

__global__ void compute_verification(double **A, int n, double *sum, double *A_37_47) {
    __shared__ double block_sum[256];
    int tid = threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double current_ele = 0.0;

    if (row < n && col < n) {
        current_ele = A[row][col];
        if (row == 37 && col == 47) {
        *A_37_47 += A[row][col];
    }
    }

    block_sum[tid] = current_ele;
    __syncthreads();

    // Parallel reduction to calculate block sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            block_sum[tid] += block_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *sum += block_sum[0];
    }
}

int main(int argc, char** argv) {
    const int t = 10;
    int n = atoi(argv[1]);

    double **A, **Ao;
    double **d_A, **d_Ao;
    double *d_sum, *d_A_37_47;
    double h_sum, h_A_37_47;

    // Initialize Matrix
    A = (double **)malloc(n * sizeof(double *));
    Ao = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; ++i) {
        A[i] = (double *)malloc(n * sizeof(double));
        Ao[i] = (double *)malloc(n * sizeof(double));
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i][j] = pow((1 + cos(2 * i) + sin(j)), 2);
        }
    }

    // Allocate memory on the GPU
    cudaMalloc(&d_A, n * sizeof(double *));
    cudaMalloc(&d_Ao, n * sizeof(double *));
    cudaMalloc(&d_sum, sizeof(double));
    cudaMalloc(&d_A_37_47, sizeof(double));
    cudaMemcpy(d_A, A, n * sizeof(double *), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    dim3 blockDim(32, 32); // Assume warp size as (32, 32)
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x,
                   (n + blockDim.y - 1) / blockDim.y);
                   
    // for (int iter = 0; iter < t; ++iter) {
    //     cudaMemcpy(d_Ao, d_A, n * sizeof(double *), cudaMemcpyDeviceToDevice);
    //     matrix_update<<<gridDim, blockDim>>>(d_A, d_Ao, n);
    // }

    // Compute verification values
    cudaMemset(d_sum, 0, sizeof(double));
    dim3 blockDim_ver(256);
    dim3 gridDim_ver((n * n + 255) / 256);
    compute_verification<<<gridDim_ver, blockDim_ver>>>(d_A, n, d_sum, d_A_37_47);
    
    // Stop timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_A_37_47, d_A_37_47, sizeof(double), cudaMemcpyDeviceToHost);
    
    std::cout << "n = " << n << ", Time elapsed = " << milliseconds << " ms\n";
    std::cout << "Sum of all entries = " << h_sum << ", A(37, 47) = " << h_A_37_47 << "\n";
    
    for (int i = 0; i < n; ++i) {
        free(A[i]);
        free(Ao[i]);
    }

    free(A);
    free(Ao);
    cudaFree(d_A);
    cudaFree(d_Ao);
    cudaFree(d_sum);
    cudaFree(d_A_37_47);
    return 0;
}
