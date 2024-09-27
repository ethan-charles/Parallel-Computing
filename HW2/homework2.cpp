#include <mpi.h>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <algorithm>

void initialize_matrix(double* A, int m, int n, int rank, int size) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            A[i * n + j] = j * std::sin(i) + i * std::cos(j) + 0.5 * i + j + 1;
        }
    }
}

double f(double x) {
    double y = x * 2.0;
    for (int i = 1; i <= 10; i++) {
        y = y + x * std::cos(y + i) / (1.5 * i);
    }
    return y;
}

void update_matrix(double* A, double* A_prev, int m, int n) {
    for (int i = 1; i < m - 1; i++) {
        for (int j = 1; j < n - 1; j++) {
            double z = (
                f(A_prev[(i - 1) * n + (j - 1)]) +
                f(A_prev[(i - 1) * n + (j + 1)]) +
                f(A_prev[(i + 1) * n + (j - 1)]) +
                f(A_prev[(i + 1) * n + (j + 1)]) +
                f(A_prev[i * n + j])
            ) / 5.0;
            A[i * n + j] = std::max(0.25, std::min(30.0, z));
        }
    }
}

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int m = /* 设置矩阵的行数 */;
    int n = /* 设置矩阵的列数 */;
    int p = /* 进程数（核心数） */;

    // 假设 p 是完美平方数
    int local_m = m / static_cast<int>(std::sqrt(p));
    double* A = new double[local_m * n];
    double* A_prev = new double[local_m * n];

    initialize_matrix(A, local_m, n, rank, size);

    MPI_Barrier(MPI_COMM_WORLD);  // 同步所有进程
    double start_time = MPI_Wtime();

    for (int iter = 0; iter < 10; iter++) {
        std::memcpy(A_prev, A, local_m * n * sizeof(double));
        update_matrix(A, A_prev, local_m, n);

        // 使用 MPI 进行边界交换，确保所有进程更新边界值
        // ... 边界交换代码 ...
        MPI_Barrier(MPI_COMM_WORLD);
    }

    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;

    // 计算验证值（总和和平方和）
    double local_sum = 0.0, local_sum_sq = 0.0;
    for (int i = 0; i < local_m * n; i++) {
        local_sum += A[i];
        local_sum_sq += A[i] * A[i];
    }

    double global_sum = 0.0, global_sum_sq = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sum_sq, &global_sum_sq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::printf("Elapsed time: %f seconds\n", elapsed_time);
        std::printf("Verification - Sum: %f, Sum of squares: %f\n", global_sum, global_sum_sq);
    }

    delete[] A;
    delete[] A_prev;
    MPI_Finalize();
    return 0;
}
