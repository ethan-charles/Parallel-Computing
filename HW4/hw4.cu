// Yichen Lu nechy@umich.edu
#include <iostream>
#include <cmath>

using namespace std;

__global__ void matrix_update(double *A, double *Ao, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i > 0 && i < n - 1 && j > 0 && j < n - 1) {
        // Find the elements up, down, left and right
        double neighbors[4] = {Ao[(i + 1) * n + (j + 1)], Ao[(i + 1) * n + (j - 1)], Ao[(i - 1) * n + (j + 1)], Ao[(i - 1) * n + (j - 1)]};

        // Bubble sort
        for (int m = 0; m < 3; ++m) {
            for (int n = m + 1; n < 4; ++n) {
                if (neighbors[m] > neighbors[n]) {
                    double temp = neighbors[m];
                    neighbors[m] = neighbors[n];
                    neighbors[n] = temp;
                }
            }
        }

        A[i * n + j] = Ao[i * n + j] + neighbors[1]; // Use the second smallest value
    }
}

__global__ void compute_verification(double *A, int n, double *sum, double *A_37_47) {
    __shared__ double block_sum[1024];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Initial block sum
    block_sum[tid] = 0.0;
    __syncthreads();

    while (idx < n * n) {
        int row = idx / n;
        int col = idx % n;
        double current_ele = A[idx];

        // Take out A 37 47
        if (row == 37 && col == 47) {
            *A_37_47 = current_ele;
        }

        block_sum[tid] += current_ele;
        idx += stride;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            block_sum[tid] += block_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        sum[blockIdx.x] = block_sum[0]; // Keep every block sum in sum[1024]
    }
}

__global__ void finalize_sum(double *block, double *block_sum, int n) {
    extern __shared__ double shared_data[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    shared_data[tid] = (i < n) ? block[i] : 0.0;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && (i + stride) < n) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        block_sum[blockIdx.x] = shared_data[0]; //Keep every block to sum until only remain 1 block
    }
}


int main(int argc, char** argv) {
    const int t = 10;
    int n = atoi(argv[1]);

    double *A, *Ao;
    double *d_A, *d_Ao;
    double *d_sum, *d_A_37_47;
    double h_total_sum, h_A_37_47;

    // Initialize A and Ao
    A = (double *)malloc(n * n * sizeof(double));
    Ao = (double *)malloc(n * n * sizeof(double));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i * n + j] = pow((1 + cos(2 * i) + sin(j)), 2);
        }
    }

    if (cudaMalloc(&d_A, n * n * sizeof(double)) != cudaSuccess) {
        cout << "Could not allocate d_A" << endl;
    }
    if (cudaMalloc(&d_Ao, n * n * sizeof(double)) != cudaSuccess) {
        cout << "Could not allocate d_Ao" << endl;
    }
    if (cudaMalloc(&d_A_37_47, sizeof(double)) != cudaSuccess) {
        cout << "Could not allocate d_A_37_47" << endl;
    }

    cudaMemset(d_A_37_47, 0, sizeof(double));

    if (cudaMemcpy(d_A, A, n * n * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
        cout << "Could not copy A into d_A" << endl;
    }

    // Start clock
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    dim3 blockDim(32, 32);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

    // Update
    for (int iter = 0; iter < t; ++iter) {
        cudaMemcpy(d_Ao, d_A, n * n * sizeof(double), cudaMemcpyDeviceToDevice);
        matrix_update<<<gridDim, blockDim>>>(d_A, d_Ao, n);
    }

    // Verification
    int block_size = 1024;
    int num_block = (n * n + block_size - 1) / block_size;
    dim3 blockDim_ver(block_size);
    dim3 gridDim_ver(num_block);

    if (cudaMalloc(&d_sum, num_block * sizeof(double)) != cudaSuccess) {
        cout << "Could not allocate d_sum" << endl;
    }

    compute_verification<<<gridDim_ver, blockDim_ver>>>(d_A, n, d_sum, d_A_37_47);
    cudaMemcpy(&h_A_37_47, d_A_37_47, sizeof(double), cudaMemcpyDeviceToHost); // Get A 37 47

    double *d_block_sum = NULL;

    while (num_block > 1) {
        int current_blocks = (num_block + 1023) / 1024;
        if (cudaMalloc(&d_block_sum, current_blocks * sizeof(double)) != cudaSuccess) {
            cout << "Could not allocate d_block_sum" << endl;
        }

        finalize_sum<<<current_blocks, 1024, 1024 * sizeof(double)>>>(d_sum, d_block_sum, num_block);

        cudaFree(d_sum);
        d_sum = d_block_sum;
        num_block = current_blocks;
        d_block_sum = NULL;
    }

    cudaMemcpy(&h_total_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost); // Get total sum
    
    // Stop clock
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "n = " << n << ", Time elapsed = " << milliseconds << " ms\n";
    cout << ", Sum of all entries = " << h_total_sum << ", A(37, 47) = " << h_A_37_47 << "\n";

    free(A);
    free(Ao);
    cudaFree(d_A);
    cudaFree(d_Ao);
    cudaFree(d_sum);
    cudaFree(d_A_37_47);
    return 0;
}
