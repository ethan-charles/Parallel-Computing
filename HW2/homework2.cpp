#include <mpi.h>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <algorithm>

// 使用命名空间
using namespace std;

// 定义矩阵尺寸和迭代次数
#define ITERATIONS 10

// 初始化矩阵
void initialize_matrix(double* A, int m, int n, int global_row_offset, int global_col_offset) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int global_i = global_row_offset + i;
            int global_j = global_col_offset + j;
             A[i * n + j] = global_j * sin(global_i) + global_i * cos(global_j) + sqrt(global_i + global_j + 1);
        }
    }
}

// 定义函数 f(x)
double f(double x) {
    double y = x * 2.0;
    for (int i = 1; i <= 10; i++) {
        y = y + x * cos(y + i) / pow(1.5, i);
    }
    return y;
}

// 更新矩阵
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
            A[i * n + j] = max(-25.0, min(30.0, z));
        }
    }
}

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 矩阵全局尺寸（根据题目要求设置）
    int m = 2000;  // 或 1000
    int n = 500;   // 或 4000

    int p = 4;  // 进程数
    int q = static_cast<int>(sqrt(p));  // 假设 p 是完美平方数

    // 每个进程负责的矩阵块尺寸
    int local_m = m / q;
    int local_n = n / q;

    // 处理不能整除的情况
    int extra_m = m % q;
    int extra_n = n % q;

    // 计算进程在进程网格中的坐标
    int row_rank = rank / q;
    int col_rank = rank % q;

    // 如果有多余的行或列，调整本地矩阵尺寸
    int start_row = row_rank * local_m + min(row_rank, extra_m);
    int start_col = col_rank * local_n + min(col_rank, extra_n);

    //Add the first kernel to 1 if there is extra
    if (row_rank < extra_m) {
        local_m ++;
    }
    if (col_rank < extra_n) {
        local_n ++;
    }

    // 分配矩阵内存，包含外部边界用于交换
    int extended_m = local_m + 2;  // 上下各增加一行
    int extended_n = local_n + 2;  // 左右各增加一列

    double* A = new double[extended_m * extended_n];
    double* A_prev = new double[extended_m * extended_n];

    // 初始化矩阵（注意全局索引偏移）
    initialize_matrix(A, local_m, local_n, start_row, start_col);

    // 初始化边界为固定值
    for (int i = 0; i < extended_m; i++) {
        A[i * extended_n] = A_prev[i * extended_n] = A[i * extended_n + 1];  // 左边界
        A[i * extended_n + extended_n - 1] = A_prev[i * extended_n + extended_n - 1] = A[i * extended_n + extended_n - 2];  // 右边界
    }
    for (int j = 0; j < extended_n; j++) {
        A[j] = A_prev[j] = A[extended_n + j];  // 上边界
        A[(extended_m - 1) * extended_n + j] = A_prev[(extended_m - 1) * extended_n + j] = A[(extended_m - 2) * extended_n + j];  // 下边界
    }

    MPI_Barrier(MPI_COMM_WORLD);  // 同步所有进程
    double start_time = MPI_Wtime();

    // 创建自定义的数据类型，便于发送和接收列数据
    // MPI_Datatype column_type;
    // MPI_Type_vector(local_m, 1, extended_n, MPI_DOUBLE, &column_type);
    // MPI_Type_commit(&column_type);

    for (int iter = 0; iter < ITERATIONS; iter++) {
        memcpy(A_prev, A, extended_m * extended_n * sizeof(double));

        // 边界交换
        // 定义相邻进程的 rank，如果不存在则设为 MPI_PROC_NULL
        int up_rank = (row_rank > 0) ? rank - q : MPI_PROC_NULL;
        int down_rank = (row_rank < q - 1) ? rank + q : MPI_PROC_NULL;
        int left_rank = (col_rank > 0) ? rank - 1 : MPI_PROC_NULL;
        int right_rank = (col_rank < q - 1) ? rank + 1 : MPI_PROC_NULL;

        // 水平交换（上下）
        // 发送第 1 行，接收上面的行
        // 发送第 1 行，接收下面的行
        MPI_Sendrecv(
            &A_prev[1 * extended_n + 1], local_n, MPI_DOUBLE, up_rank, 0,
            &A_prev[(local_m + 1) * extended_n + 1], local_n, MPI_DOUBLE, down_rank, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );

        // 发送第 local_m 行，接收上面的行
        MPI_Sendrecv(
            &A_prev[local_m * extended_n + 1], local_n, MPI_DOUBLE, down_rank, 0,
            &A_prev[0 * extended_n + 1], local_n, MPI_DOUBLE, up_rank, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );

        

        double* send_left = new double[local_m];
        double* recv_left = new double[local_m];
        double* send_right = new double[local_m];
        double* recv_right = new double[local_m];

        // 提取要发送的左边界列数据（第 1 列）
        for (int i = 0; i < local_m; i++) {
            send_left[i] = A_prev[(i + 1) * extended_n + 1];
        }

        // 提取要发送的右边界列数据（第 local_n 列）
        for (int i = 0; i < local_m; i++) {
            send_right[i] = A_prev[(i + 1) * extended_n + local_n];
        }

        // 发送和接收左边界列数据
        MPI_Sendrecv(
            send_left, local_m, MPI_DOUBLE, left_rank, 1,
            recv_right, local_m, MPI_DOUBLE, right_rank, 1,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );

        // 发送和接收右边界列数据
        MPI_Sendrecv(
            send_right, local_m, MPI_DOUBLE, right_rank, 1,
            recv_left, local_m, MPI_DOUBLE, left_rank, 1,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );

        // 将接收到的左边界列数据复制回矩阵（存储在第 0 列）
        for (int i = 0; i < local_m; i++) {
            A_prev[(i + 1) * extended_n + 0] = recv_left[i];
        }

        // 将接收到的右边界列数据复制回矩阵（存储在第 extended_n - 1 列）
        for (int i = 0; i < local_m; i++) {
            A_prev[(i + 1) * extended_n + extended_n - 1] = recv_right[i];
        }

        delete[] send_left;
        delete[] recv_left;
        delete[] send_right;
        delete[] recv_right;

        // 角点交换（对角方向）
        // 定义对角方向的相邻进程的 rank，如果不存在则设为 MPI_PROC_NULL
        int up_left_rank = (up_rank != MPI_PROC_NULL && left_rank != MPI_PROC_NULL) ? up_rank - 1 : MPI_PROC_NULL;
        int up_right_rank = (up_rank != MPI_PROC_NULL && right_rank != MPI_PROC_NULL) ? up_rank + 1 : MPI_PROC_NULL;
        int down_left_rank = (down_rank != MPI_PROC_NULL && left_rank != MPI_PROC_NULL) ? down_rank - 1 : MPI_PROC_NULL;
        int down_right_rank = (down_rank != MPI_PROC_NULL && right_rank != MPI_PROC_NULL) ? down_rank + 1 : MPI_PROC_NULL;

        // 对角方向边界交换
        double send_value, recv_value;
        MPI_Status status;

        // 左上角
        if (up_left_rank != MPI_PROC_NULL) {
            send_value = A_prev[1 * extended_n + 1];
            MPI_Sendrecv(
                &send_value, 1, MPI_DOUBLE, up_left_rank, 2,
                &recv_value, 1, MPI_DOUBLE, up_left_rank, 2,
                MPI_COMM_WORLD, &status
            );
            A_prev[0 * extended_n + 0] = recv_value;
        }

        // 右上角
        if (up_right_rank != MPI_PROC_NULL) {
            send_value = A_prev[1 * extended_n + local_n];
            MPI_Sendrecv(
                &send_value, 1, MPI_DOUBLE, up_right_rank, 2,
                &recv_value, 1, MPI_DOUBLE, up_right_rank, 2,
                MPI_COMM_WORLD, &status
            );
            A_prev[0 * extended_n + extended_n - 1] = recv_value;
        }

        // 左下角
        if (down_left_rank != MPI_PROC_NULL) {
            send_value = A_prev[local_m * extended_n + 1];
            MPI_Sendrecv(
                &send_value, 1, MPI_DOUBLE, down_left_rank, 2,
                &recv_value, 1, MPI_DOUBLE, down_left_rank, 2,
                MPI_COMM_WORLD, &status
            );
            A_prev[(local_m + 1) * extended_n + 0] = recv_value;
        }

        // 右下角
        if (down_right_rank != MPI_PROC_NULL) {
            send_value = A_prev[local_m * extended_n + local_n];
            MPI_Sendrecv(
                &send_value, 1, MPI_DOUBLE, down_right_rank, 2,
                &recv_value, 1, MPI_DOUBLE, down_right_rank, 2,
                MPI_COMM_WORLD, &status
            );
            A_prev[(local_m + 1) * extended_n + extended_n - 1] = recv_value;
        }



        // 更新矩阵
        update_matrix(A, A_prev, extended_m, extended_n);

        // 同步进程
        MPI_Barrier(MPI_COMM_WORLD);
    }

    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;

    // 计算验证值（总和和平方和）
    double local_sum = 0.0, local_sum_sq = 0.0;
    for (int i = 1; i <= local_m; i++) {
        for (int j = 1; j <= local_n; j++) {
            double val = A[i * extended_n + j];
            local_sum += val;
            local_sum_sq += val * val;
        }
    }

    double global_sum = 0.0, global_sum_sq = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sum_sq, &global_sum_sq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Elapsed time: %f seconds\n", elapsed_time);
        printf("Verification - Sum: %f, Sum of squares: %f\n", global_sum, global_sum_sq);
    }

    // 释放资源
    delete[] A;
    delete[] A_prev;
    MPI_Finalize();
    return 0;
}
