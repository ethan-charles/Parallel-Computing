#include <vector>
#include <cudnn.h>
#include <iostream>

// Utility function for cuDNN error handling
void checkCUDNN(cudnnStatus_t status) {
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << "cuDNN Error: " << cudnnGetErrorString(status) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Forward mapping (H) using cuDNN
void H(const std::vector<float> &input, const std::vector<float> &psf, std::vector<float> &output,
       int inputX, int inputY, int inputZ,
       int psfX, int psfY, int psfZ) {
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    // Tensor descriptors
    cudnnTensorDescriptor_t inputDesc, outputDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;

    checkCUDNN(cudnnCreateTensorDescriptor(&inputDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&outputDesc));
    checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

    checkCUDNN(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, inputY * inputZ, inputX));
    checkCUDNN(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, inputY * inputZ, inputX));
    checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 1, psfY * psfZ, psfX));

    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, 0, 0, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));

    // Allocate device memory
    float *d_input, *d_psf, *d_output;
    size_t inputSize = inputX * inputY * inputZ * sizeof(float);
    size_t psfSize = psfX * psfY * psfZ * sizeof(float);
    size_t outputSize = inputX * inputY * inputZ * sizeof(float);

    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_psf, psfSize);
    cudaMalloc(&d_output, outputSize);

    cudaMemcpy(d_input, input.data(), inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_psf, psf.data(), psfSize, cudaMemcpyHostToDevice);

    // Determine workspace size
    size_t workspaceSize;
    cudnnConvolutionFwdAlgo_t algo;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn, inputDesc, filterDesc, convDesc, outputDesc,
                                                    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, inputDesc, filterDesc, convDesc, outputDesc, algo, &workspaceSize));

    void *workspace;
    cudaMalloc(&workspace, workspaceSize);

    // Perform convolution
    const float alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnConvolutionForward(cudnn, &alpha, inputDesc, d_input, filterDesc, d_psf, convDesc, algo, workspace, workspaceSize, &beta, outputDesc, d_output));

    cudaMemcpy(output.data(), d_output, outputSize, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_psf);
    cudaFree(d_output);
    cudaFree(workspace);

    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroyTensorDescriptor(outputDesc);
    cudnnDestroyFilterDescriptor(filterDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroy(cudnn);
}

// Adjoint mapping (HT) using cuDNN
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
