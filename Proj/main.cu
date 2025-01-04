#include "FISTANet.cu"
#include "conv_helper.cu"
#include <vector>
#include <cmath>
#include <iostream>
#include <nifti/nifti1_io.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

// Dataset and configuration parameters
const std::string DATA_DIR = "./data";
const int NUM_FILE = 100;
const int BATCH_SIZE = 10;
const std::string PSF_DIR = "./psf_data";

// Custom Dataset Loader
class CustomDataset {
private:
    std::string data_dir;
    int num_files;

public:
    CustomDataset(const std::string &dir, int files) : data_dir(dir), num_files(files) {}

    size_t size() const { return num_files; }

    void loadData(int idx, std::vector<float> &scanning_data, std::vector<float> &ground_truth, int inputX, int inputY, int inputZ) {
        char sd_path[256], gt_path[256];
        snprintf(sd_path, sizeof(sd_path), "%s/mimic_b80p20_scanning_data%d.nii", data_dir.c_str(), idx + 1);
        snprintf(gt_path, sizeof(gt_path), "%s/mimic_b80p20_ground_truth%d.nii", data_dir.c_str(), idx + 1);

        nifti_image *sd_img = nifti_image_read(sd_path, 1);
        nifti_image *gt_img = nifti_image_read(gt_path, 1);

        if (!sd_img || !gt_img) {
            throw std::runtime_error("Failed to load image files");
        }

        scanning_data.assign(reinterpret_cast<float *>(sd_img->data),
                             reinterpret_cast<float *>(sd_img->data) + inputX * inputY * inputZ);
        ground_truth.assign(reinterpret_cast<float *>(gt_img->data),
                            reinterpret_cast<float *>(gt_img->data) + inputX * inputY * inputZ);

        nifti_image_free(sd_img);
        nifti_image_free(gt_img);
    }
};

// Load PSF data
void loadPSF(const std::string &psf_file, std::vector<float> &psf, int psfX, int psfY, int psfZ, int batch_size) {
    nifti_image *psf_img = nifti_image_read(psf_file.c_str(), 1);
    if (!psf_img) {
        throw std::runtime_error("Failed to load PSF file");
    }

    std::vector<float> psf_raw(reinterpret_cast<float *>(psf_img->data),
                               reinterpret_cast<float *>(psf_img->data) + psfX * psfY * psfZ);

    psf.resize(batch_size * psfX * psfY * psfZ);
    for (int i = 0; i < batch_size; ++i) {
        std::copy(psf_raw.begin(), psf_raw.end(), psf.begin() + i * psfX * psfY * psfZ);
    }

    nifti_image_free(psf_img);
}

// Training function with multi-GPU support
void train(FISTANet &model, CustomDataset &dataset, const std::vector<float> &psf,
           int inputX, int inputY, int inputZ, int epochs, float learning_rate, int num_gpus) {
    int total_files = dataset.size();
    int batch_count = total_files / BATCH_SIZE;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;

        for (int batch_idx = 0; batch_idx < batch_count; ++batch_idx) {
            std::vector<std::vector<float>> batch_data(BATCH_SIZE, std::vector<float>(inputX * inputY * inputZ));
            std::vector<std::vector<float>> batch_labels(BATCH_SIZE, std::vector<float>(inputX * inputY * inputZ));

            for (int i = 0; i < BATCH_SIZE; ++i) {
                dataset.loadData(batch_idx * BATCH_SIZE + i, batch_data[i], batch_labels[i], inputX, inputY, inputZ);
            }

            #pragma omp parallel for num_threads(num_gpus)
            for (int gpu_idx = 0; gpu_idx < num_gpus; ++gpu_idx) {
                cudaSetDevice(gpu_idx);

                int start_idx = gpu_idx * BATCH_SIZE / num_gpus;
                int end_idx = (gpu_idx + 1) * BATCH_SIZE / num_gpus;

                for (int i = start_idx; i < end_idx; ++i) {
                    std::vector<float> output;

                    // Forward pass
                    model.forward(batch_data[i], psf, output);

                    // Compute loss (MSE)
                    float loss = 0.0f;
                    for (size_t j = 0; j < output.size(); ++j) {
                        float diff = output[j] - batch_labels[i][j];
                        loss += diff * diff;
                    }
                    total_loss += loss / output.size();

                    // Backpropagation and parameter update would go here
                }
            }
        }
        std::cout << "Epoch " << epoch + 1 << "/" << epochs << " - Loss: " << total_loss / total_files << std::endl;
    }
}

int main() {
    // Example dimensions
    int inputX = 32, inputY = 32, inputZ = 32;
    int psfX = 5, psfY = 5, psfZ = 5;
    int num_layers = 5;
    int num_gpus = 2; // Specify the number of GPUs

    // Initialize FISTANet
    FISTANet model(num_layers, inputX, inputY, inputZ, psfX, psfY, psfZ);

    // Load PSF data
    std::vector<float> psf;
    loadPSF(PSF_DIR, psf, psfX, psfY, psfZ, BATCH_SIZE);

    // Initialize dataset
    CustomDataset dataset(DATA_DIR, NUM_FILE);

    // Train the model
    train(model, dataset, psf, inputX, inputY, inputZ, 10, 0.01f, num_gpus);

    return 0;
}
