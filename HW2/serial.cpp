#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <ctime>

using namespace std;
#define ITERATIONS 10

double f(double x) {
    // 定义函数 f(x)
    double y = x * 2.0;
    for (int i = 1; i <= 10; i++) {
        y = y + x * cos(y + i) / pow(1.5, i);
    }
    return y;
}
void initialize_matrix(double* A, int m, int n) {
    // 初始化矩阵 A，尺寸为 m x n
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            A[i * n + j] = j * sin(i) + i * cos(j) + sqrt(i + j + 1);
        }
    }
}
void update_matrix(double* A, double* A_prev, int m, int n) {
    // 更新矩阵 A，使用前一轮的值 A_prev
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
int main() {
    // 矩阵全局尺寸
    int m = 2000;  // 或 1000
    int n = 500;   // 或 4000

    // 分配矩阵内存
    double* A = new double[m * n];
    double* A_prev = new double[m * n];

    // 初始化矩阵
    initialize_matrix(A, m, n);

    // 记录开始时间
    clock_t start_time = clock();

    // 迭代更新矩阵
    for (int iter = 0; iter < ITERATIONS; iter++) {
        // 复制当前矩阵到 A_prev
        memcpy(A_prev, A, m * n * sizeof(double));

        // 更新矩阵
        update_matrix(A, A_prev, m, n);
    }

    // 记录结束时间
    clock_t end_time = clock();
    double elapsed_time = double(end_time - start_time) / CLOCKS_PER_SEC;

    // 计算验证值（总和和平方和）
    double sum = 0.0, sum_sq = 0.0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double val = A[i * n + j];
            sum += val;
            sum_sq += val * val;
        }
    }

    // 输出结果
    cout << "Elapsed time: " << elapsed_time << " seconds" << endl;
    cout << "Sum of all entries: " << sum << endl;
    cout << "Sum of squares of entries: " << sum_sq << endl;

    // 释放资源
    delete[] A;
    delete[] A_prev;

    return 0;
}
