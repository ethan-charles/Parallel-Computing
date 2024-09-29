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
    int extended_m = local_m + 2;  // 上下各增加一行#include <mpi.h>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <algorithm>

// 使用命名空间
using namespace std;

// 定义矩阵尺寸和迭代次数
#define ITERATIONS 10

// 定义函数 f(x)
double f(double x) {
    double y = x * 2.0;
    for (int i = 1; i <= 10; i++) {
        y = y + x * cos(y + i) / pow(1.5, i);
    }
    return y;
}

// 更新矩阵
void update_matrix(double* A, double* A_prev, int m, int n, int rank_row, int rank_col, int q) {
    for (int i = 1; i < m - 1; i++) {
        for (int j = 1; j < n - 1; j++) {
            // 边界条件：全局矩阵的边界元素保持不变
            if ((i == 1 && rank_row == 0) || (i == m - 2 && rank_row == q - 1) ||
                (j == 1 && rank_col == 0) || (j == n - 2 && rank_col == q - 1)) {
                A[i * n + j] = A_prev[i * n + j]; // 保持边界元素不变
            } else {
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
}

// 初始化矩阵，包括幽灵层和边界条件处理
void initialize_matrix(double* A, int extended_m, int extended_n, int global_row_offset, int global_col_offset, int local_m, int local_n, int rank_row, int rank_col, int q) {
    // 初始化整个矩阵为0，包括幽灵层
    memset(A, 0, extended_m * extended_n * sizeof(double));

    // 遍历包括幽灵层的矩阵，初始化实际数据部分
    for (int i = -1; i <= local_m; i++) {
        for (int j = -1; j <= local_n; j++) {
            // 全局索引
            int global_i = global_row_offset + i;
            int global_j = global_col_offset + j;

            // 处理全局矩阵边界条件
            if ((i == -1 && rank_row == 0) || (i == local_m && rank_row == q - 1) ||
                (j == -1 && rank_col == 0) || (j == local_n && rank_col == q - 1)) {
                continue; // 跳过全局边界的幽灵层
            }

            // 本地矩阵索引（包含幽灵层偏移）
            int local_i = i + 1;
            int local_j = j + 1;

            A[local_i * extended_n + local_j] = global_j * sin(global_i) + global_i * cos(global_j) + sqrt(global_i + global_j + 1);
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

    int p = size;  // 进程数
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

    if (row_rank < extra_m) {
        local_m++;
    }
    if (col_rank < extra_n) {
        local_n++;
    }

    // 分配矩阵内存，包含外部边界用于交换
    int extended_m = local_m + 2;  // 上下各增加一行（幽灵层）
    int extended_n = local_n + 2;  // 左右各增加一列（幽灵层）

    double* A = new double[extended_m * extended_n];
    double* A_prev = new double[extended_m * extended_n];

    // 初始化矩阵（包括幽灵层和边界条件处理）
    initialize_matrix(A, extended_m, extended_n, start_row, start_col, local_m, local_n, row_rank, col_rank, q);

    // 初始化 A_prev
    memcpy(A_prev, A, extended_m * extended_n * sizeof(double));

    MPI_Barrier(MPI_COMM_WORLD);  // 同步所有进程
    double start_time = MPI_Wtime();

    // 在迭代循环外定义通信标签和缓冲区
    int tag_right = 0;
    int tag_left = 1;
    int tag_up = 2;
    int tag_down = 3;

    double* buf_col = new double[extended_m - 2]; // 用于列通信的缓冲区（排除幽灵层）
    double* buf_row = new double[extended_n - 2]; // 用于行通信的缓冲区（排除幽灵层）

    for (int iter = 0; iter < ITERATIONS; iter++) {
        memcpy(A_prev, A, extended_m * extended_n * sizeof(double));

        // 定义相邻进程的 rank
        int up_rank = (row_rank > 0) ? rank - q : MPI_PROC_NULL;
        int down_rank = (row_rank < q - 1) ? rank + q : MPI_PROC_NULL;
        int left_rank = (col_rank > 0) ? rank - 1 : MPI_PROC_NULL;
        int right_rank = (col_rank < q - 1) ? rank + 1 : MPI_PROC_NULL;

        // **垂直方向通信（左右）**

        // **发送右边界列到右边的进程**
        if (col_rank != q - 1) { // 非最右列进程
            for (int i = 1; i < extended_m - 1; ++i) {
                buf_col[i - 1] = A_prev[i * extended_n + local_n]; // 发送第 local_n 列
            }
            MPI_Send(buf_col, extended_m - 2, MPI_DOUBLE, right_rank, tag_right, MPI_COMM_WORLD);
        }
        // **接收左边进程发送的左边界列，存储在本地矩阵的左幽灵层**
        if (col_rank != 0) { // 非最左列进程
            MPI_Recv(buf_col, extended_m - 2, MPI_DOUBLE, left_rank, tag_right, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 1; i < extended_m - 1; ++i) {
                A_prev[i * extended_n + 0] = buf_col[i - 1]; // 接收数据存储在第 0 列（左幽灵层）
            }
        }

        // **发送左边界列到左边的进程**
        if (col_rank != 0) { // 非最左列进程
            for (int i = 1; i < extended_m - 1; ++i) {
                buf_col[i - 1] = A_prev[i * extended_n + 1]; // 发送第 1 列
            }
            MPI_Send(buf_col, extended_m - 2, MPI_DOUBLE, left_rank, tag_left, MPI_COMM_WORLD);
        }
        // **接收右边进程发送的右边界列，存储在本地矩阵的右幽灵层**
        if (col_rank != q - 1) { // 非最右列进程
            MPI_Recv(buf_col, extended_m - 2, MPI_DOUBLE, right_rank, tag_left, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 1; i < extended_m - 1; ++i) {
                A_prev[i * extended_n + extended_n - 1] = buf_col[i - 1]; // 接收数据存储在第 extended_n - 1 列（右幽灵层）
            }
        }

        // **水平方向通信（上下）**

        // **发送上边界行到上面的进程**
        if (row_rank != 0) { // 非最上行进程
            memcpy(buf_row, &A_prev[1 * extended_n + 1], (extended_n - 2) * sizeof(double)); // 发送第 1 行的数据，排除左右幽灵层
            MPI_Send(buf_row, extended_n - 2, MPI_DOUBLE, up_rank, tag_up, MPI_COMM_WORLD);
        }
        // **接收下边进程发送的下边界行，存储在本地矩阵的下幽灵层**
        if (row_rank != q - 1) { // 非最下行进程
            MPI_Recv(buf_row, extended_n - 2, MPI_DOUBLE, down_rank, tag_up, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            memcpy(&A_prev[(extended_m - 1) * extended_n + 1], buf_row, (extended_n - 2) * sizeof(double)); // 存储在第 extended_m - 1 行（下幽灵层）
        }

        // **发送下边界行到下面的进程**
        if (row_rank != q - 1) { // 非最下行进程
            memcpy(buf_row, &A_prev[(extended_m - 2) * extended_n + 1], (extended_n - 2) * sizeof(double)); // 发送第 extended_m - 2 行的数据
            MPI_Send(buf_row, extended_n - 2, MPI_DOUBLE, down_rank, tag_down, MPI_COMM_WORLD);
        }
        // **接收上边进程发送的上边界行，存储在本地矩阵的上幽灵层**
        if (row_rank != 0) { // 非最上行进程
            MPI_Recv(buf_row, extended_n - 2, MPI_DOUBLE, up_rank, tag_down, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            memcpy(&A_prev[0 * extended_n + 1], buf_row, (extended_n - 2) * sizeof(double)); // 存储在第 0 行（上幽灵层）
        }

        // **更新矩阵**
        update_matrix(A, A_prev, extended_m, extended_n, row_rank, col_rank, q);

        // **同步进程**
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // **释放缓冲区**
    delete[] buf_col;
    delete[] buf_row;

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
        printf("Sum of all entries: %f\n", global_sum);
        printf("Sum of squares of entries: %f\n", global_sum_sq);
    }

    // 释放资源
    delete[] A;
    delete[] A_prev;
    MPI_Finalize();
    return 0;
}

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
    
    

    // 在迭代循环外定义通信标签和缓冲区
    int tag_right = 0;
    int tag_left = 1;
    int tag_up = 2;
    int tag_down = 3;
    
    double* buf_col = new double[extended_m]; // 用于列通信的缓冲区（包含上下边界）
    double* buf_row = new double[extended_n]; // 用于行通信的缓冲区（包含左右边界）
    
    for (int iter = 0; iter < ITERATIONS; iter++) {
        memcpy(A_prev, A, extended_m * extended_n * sizeof(double));
    
        // 定义相邻进程的 rank
        int up_rank = (row_rank > 0) ? rank - q : MPI_PROC_NULL;
        int down_rank = (row_rank < q - 1) ? rank + q : MPI_PROC_NULL;
        int left_rank = (col_rank > 0) ? rank - 1 : MPI_PROC_NULL;
        int right_rank = (col_rank < q - 1) ? rank + 1 : MPI_PROC_NULL;
    
        // **垂直方向通信（左右）**
    
        // **发送右边界列到右边的进程**
        if (col_rank != q - 1) { // 非最右列进程
            for (int i = 0; i < extended_m; ++i) {
                buf_col[i] = A_prev[i * extended_n + local_n]; // 发送第 local_n 列
            }
            MPI_Send(buf_col, extended_m, MPI_DOUBLE, right_rank, tag_right, MPI_COMM_WORLD);
        }
        // **接收左边进程发送的左边界列，存储在本地矩阵的左边界**
        if (col_rank != 0) { // 非最左列进程
            MPI_Recv(buf_col, extended_m, MPI_DOUBLE, left_rank, tag_right, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < extended_m; ++i) {
                A_prev[i * extended_n + 0] = buf_col[i]; // 接收数据存储在第 0 列
            }
        }
    
        // **发送左边界列到左边的进程**
        if (col_rank != 0) { // 非最左列进程
            for (int i = 0; i < extended_m; ++i) {
                buf_col[i] = A_prev[i * extended_n + 1]; // 发送第 1 列
            }
            MPI_Send(buf_col, extended_m, MPI_DOUBLE, left_rank, tag_left, MPI_COMM_WORLD);
        }
        // **接收右边进程发送的右边界列，存储在本地矩阵的右边界**
        if (col_rank != q - 1) { // 非最右列进程
            MPI_Recv(buf_col, extended_m, MPI_DOUBLE, right_rank, tag_left, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < extended_m; ++i) {
                A_prev[i * extended_n + extended_n - 1] = buf_col[i]; // 接收数据存储在第 extended_n - 1 列
            }
        }
    
        // **水平方向通信（上下）**
    
        // **发送上边界行到上面的进程**
        if (row_rank != 0) { // 非最上行进程
            memcpy(buf_row, &A_prev[1 * extended_n], extended_n * sizeof(double)); // 发送第 1 行
            MPI_Send(buf_row, extended_n, MPI_DOUBLE, up_rank, tag_up, MPI_COMM_WORLD);
        }
        // **接收下边进程发送的下边界行，存储在本地矩阵的下边界**
        if (row_rank != q - 1) { // 非最下行进程
            MPI_Recv(buf_row, extended_n, MPI_DOUBLE, down_rank, tag_up, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            memcpy(&A_prev[(extended_m - 1) * extended_n], buf_row, extended_n * sizeof(double)); // 存储在第 extended_m - 1 行
        }
    
        // **发送下边界行到下面的进程**
        if (row_rank != q - 1) { // 非最下行进程
            memcpy(buf_row, &A_prev[local_m * extended_n], extended_n * sizeof(double)); // 发送第 local_m 行
            MPI_Send(buf_row, extended_n, MPI_DOUBLE, down_rank, tag_down, MPI_COMM_WORLD);
        }
        // **接收上边进程发送的上边界行，存储在本地矩阵的上边界**
        if (row_rank != 0) { // 非最上行进程
            MPI_Recv(buf_row, extended_n, MPI_DOUBLE, up_rank, tag_down, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            memcpy(&A_prev[0], buf_row, extended_n * sizeof(double)); // 存储在第 0 行
        }
    
        // **更新矩阵**
        update_matrix(A, A_prev, extended_m, extended_n);
    
        // **同步进程**
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    // **释放缓冲区**
    delete[] buf_col;
    delete[] buf_row;

    
    double end_time = MPI_Wtime();
        
    
    

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
        double elapsed_time = end_time - start_time;
        printf("Elapsed time: %f seconds\n", elapsed_time);
        printf("Verification - Sum: %f, Sum of squares: %f\n", global_sum, global_sum_sq);
    }

    // 释放资源
    delete[] A;
    delete[] A_prev;
    MPI_Finalize();
    return 0;
}
