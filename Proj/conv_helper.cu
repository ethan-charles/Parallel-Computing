#include <vector>

// CUDA Kernel for 3D convolution
__global__ void convolution3d(const float *input, const float *psf, float *output,
                              int inputX, int inputY, int inputZ,
                              int psfX, int psfY, int psfZ) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < inputX && y < inputY && z < inputZ) {
        float value = 0.0f;

        for (int i = 0; i < psfX; ++i) {
            for (int j = 0; j < psfY; ++j) {
                for (int k = 0; k < psfZ; ++k) {
                    int xi = x + i - psfX / 2;
                    int yj = y + j - psfY / 2;
                    int zk = z + k - psfZ / 2;

                    if (xi >= 0 && xi < inputX && yj >= 0 && yj < inputY && zk >= 0 && zk < inputZ) {
                        value += input[(zk * inputY + yj) * inputX + xi] *
                                 psf[(k * psfY + j) * psfX + i];
                    }
                }
            }
        }
        output[(z * inputY + y) * inputX + x] = value;
    }
}

// Forward mapping (H)
void H(const std::vector<float> &input, const std::vector<float> &psf, std::vector<float> &output,
       int inputX, int inputY, int inputZ,
       int psfX, int psfY, int psfZ) {
    size_t inputSize = inputX * inputY * inputZ * sizeof(float);
    size_t psfSize = psfX * psfY * psfZ * sizeof(float);
    size_t outputSize = inputX * inputY * inputZ * sizeof(float);

    float *d_input, *d_psf, *d_output;
    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_psf, psfSize);
    cudaMalloc(&d_output, outputSize);

    cudaMemcpy(d_input, input.data(), inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_psf, psf.data(), psfSize, cudaMemcpyHostToDevice);

    dim3 blockSize(8, 8, 8);
    dim3 gridSize((inputX + blockSize.x - 1) / blockSize.x,
                  (inputY + blockSize.y - 1) / blockSize.y,
                  (inputZ + blockSize.z - 1) / blockSize.z);

    convolution3d<<<gridSize, blockSize>>>(d_input, d_psf, d_output, inputX, inputY, inputZ, psfX, psfY, psfZ);
    cudaMemcpy(output.data(), d_output, outputSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_psf);
    cudaFree(d_output);
}

// Adjoint mapping (HT)
void HT(const std::vector<float> &input, const std::vector<float> &psf, std::vector<float> &output,
        int inputX, int inputY, int inputZ,
        int psfX, int psfY, int psfZ) {
    // Flip PSF for adjoint mapping
    std::vector<float> flippedPsf(psfX * psfY * psfZ);
    for (int i = 0; i < psfX; ++i) {
        for (int j = 0; j < psfY; ++j) {
            for (int k = 0; k < psfZ; ++k) {
                flippedPsf[(k * psfY + j) * psfX + i] =
                    psf[((psfZ - 1 - k) * psfY + (psfY - 1 - j)) * psfX + (psfX - 1 - i)];
            }
        }
    }

    H(input, flippedPsf, output, inputX, inputY, inputZ, psfX, psfY, psfZ);
}
