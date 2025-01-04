#include "fistanet_cuda.cu"
#include <vector>
#include <cmath>

// Define BasicBlock class optimized with CUDA
class BasicBlock {
private:
    float* lambda_step;
    float* soft_thr;
    float* conv_D;
    float* conv1_forward;
    float* conv2_forward;
    float* conv1_backward;
    float* conv2_backward;
    float* conv_G;

    int batch_size;
    int inputX, inputY, inputZ;
    int psfX, psfY, psfZ;

public:
    BasicBlock(int batch, int inX, int inY, int inZ, int psfSizeX, int psfSizeY, int psfSizeZ)
        : batch_size(batch), inputX(inX), inputY(inY), inputZ(inZ),
          psfX(psfSizeX), psfY(psfSizeY), psfZ(psfSizeZ) {

        cudaMalloc(&lambda_step, sizeof(float));
        cudaMalloc(&soft_thr, sizeof(float));
        cudaMalloc(&conv_D, 32 * 1 * 3 * 3 * 3 * sizeof(float));
        cudaMalloc(&conv1_forward, 32 * 32 * 3 * 3 * 3 * sizeof(float));
        cudaMalloc(&conv2_forward, 32 * 32 * 3 * 3 * 3 * sizeof(float));
        cudaMalloc(&conv1_backward, 32 * 32 * 3 * 3 * 3 * sizeof(float));
        cudaMalloc(&conv2_backward, 32 * 32 * 3 * 3 * 3 * sizeof(float));
        cudaMalloc(&conv_G, 1 * 32 * 3 * 3 * 3 * sizeof(float));

        // Initialize parameters (copy to device)
        float h_lambda_step = 0.1f;
        float h_soft_thr = 0.01f;
        cudaMemcpy(lambda_step, &h_lambda_step, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(soft_thr, &h_soft_thr, sizeof(float), cudaMemcpyHostToDevice);
    }

    ~BasicBlock() {
        cudaFree(lambda_step);
        cudaFree(soft_thr);
        cudaFree(conv_D);
        cudaFree(conv1_forward);
        cudaFree(conv2_forward);
        cudaFree(conv1_backward);
        cudaFree(conv2_backward);
        cudaFree(conv_G);
    }

    void forward(float* x, const float* psf, const float* b,
                 float* output, float* symloss) {

        dim3 block(8, 8, 8);
        dim3 grid((inputX + block.x - 1) / block.x, (inputY + block.y - 1) / block.y, (inputZ + block.z - 1) / block.z);

        // HT operation using CUDA
        float* htx_output;
        float* h_output;
        cudaMalloc(&htx_output, batch_size * inputX * inputY * inputZ * sizeof(float));
        cudaMalloc(&h_output, batch_size * inputX * inputY * inputZ * sizeof(float));

        H_cuda<<<grid, block>>>(x, psf, h_output, inputX, inputY, inputZ, psfX, psfY, psfZ);

        cudaMemcpy(htx_output, h_output, batch_size * inputX * inputY * inputZ * sizeof(float), cudaMemcpyDeviceToDevice);
        HT_cuda<<<grid, block>>>(htx_output, psf, htx_output, inputX, inputY, inputZ, psfX, psfY, psfZ);

        cudaMemcpy(x, htx_output, batch_size * inputX * inputY * inputZ * sizeof(float), cudaMemcpyDeviceToDevice);

        cudaFree(htx_output);
        cudaFree(h_output);

        // Convolution and activation CUDA kernels...
        // Skipped detailed kernel implementations for brevity

        // Final prediction output calculation
        float* x_G;
        cudaMalloc(&x_G, batch_size * inputX * inputY * inputZ * sizeof(float));
        convolution_cuda<<<grid, block>>>(x, conv_G, x_G, inputX, inputY, inputZ);

        // Combine input and x_G, apply ReLU
        relu_combine_cuda<<<grid, block>>>(x, x_G, output, inputX, inputY, inputZ);

        cudaFree(x_G);
    }
};

// Define FISTANet class optimized with CUDA
class FISTANet {
private:
    int num_layers;
    int batch_size;
    int inputX, inputY, inputZ;
    int psfX, psfY, psfZ;

    std::vector<BasicBlock> layers;
    float w_rho;
    float b_rho;

public:
    FISTANet(int layers_no, int batch, int inX, int inY, int inZ, int psfSizeX, int psfSizeY, int psfSizeZ)
        : num_layers(layers_no), batch_size(batch), inputX(inX), inputY(inY), inputZ(inZ),
          psfX(psfSizeX), psfY(psfSizeY), psfZ(psfSizeZ), w_rho(0.5f), b_rho(0.0f) {

        for (int i = 0; i < num_layers; ++i) {
            layers.emplace_back(BasicBlock(batch, inX, inY, inZ, psfSizeX, psfSizeY, psfSizeZ));
        }
    }

    void forward(float* measured, float* psf, float* output, float* layers_sym) {
        float* x0;
        cudaMalloc(&x0, batch_size * inputX * inputY * inputZ * sizeof(float));
        cudaMemset(x0, 1, batch_size * inputX * inputY * inputZ * sizeof(float));

        float* xold;
        cudaMalloc(&xold, batch_size * inputX * inputY * inputZ * sizeof(float));
        cudaMemcpy(xold, x0, batch_size * inputX * inputY * inputZ * sizeof(float), cudaMemcpyDeviceToDevice);

        float* y;
        cudaMalloc(&y, batch_size * inputX * inputY * inputZ * sizeof(float));
        cudaMemcpy(y, x0, batch_size * inputX * inputY * inputZ * sizeof(float), cudaMemcpyDeviceToDevice);

        cudaMemset(layers_sym, 0, batch_size * inputX * inputY * inputZ * sizeof(float));

        for (int i = 0; i < num_layers; ++i) {
            float* xnew;
            float* layer_sym;
            cudaMalloc(&xnew, batch_size * inputX * inputY * inputZ * sizeof(float));
            cudaMalloc(&layer_sym, batch_size * inputX * inputY * inputZ * sizeof(float));

            layers[i].forward(y, psf, measured, xnew, layer_sym);

            float rho = (std::log(1 + std::exp(w_rho * i + b_rho)) - std::log(1 + std::exp(b_rho))) /
                        std::log(1 + std::exp(w_rho * i + b_rho));

            // CUDA kernel for two-step update (xnew -> y)
            two_step_update_cuda<<<grid, block>>>(xnew, xold, y, rho, inputX, inputY, inputZ);

            cudaFree(xold);
            xold = xnew;
            cudaFree(layer_sym);
        }

        cudaMemcpy(output, xold, batch_size * inputX * inputY * inputZ * sizeof(float), cudaMemcpyDeviceToDevice);

        cudaFree(x0);
        cudaFree(xold);
        cudaFree(y);
    }
};
