#include <mpi.h>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <iostream>

using namespace std;
#define ITERATIONS 10

void initialize_matrix(double* A, int m, int n, int start_row, int start_col, int rank_row, int rank_col, int q) {
    memset(A, 0, m * n * sizeof(double));

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {

            int global_i = start_row + i - 1;
            int global_j = start_col + j - 1;

            // Skip all ghost layers
            if ((i == 0 && rank_row == 0) || (i == m - 1 && rank_row == q - 1) ||
                (j == 0 && rank_col == 0) || (j == n - 1 && rank_col == q - 1)) {
                continue;
            }

            // Initialize A matrix
            A[i * n + j] = global_j * sin(global_i) + global_i * cos(global_j) + sqrt(global_i + global_j + 1);
        }
    }
}


double f(double x) {
    double y = x * 2.0;
    for (int i = 1; i <= 10; i++) {
        y = y + x * cos(y + i) / pow(1.5, i);
    }
    return y;
}


void update_matrix(double* A, double* A_prev, int m, int n, int rank_row, int rank_col, int q) {
    for (int i = 1; i < m - 1; i++) {
        for (int j = 1; j < n - 1; j++) {

            // Skip all ghost layers
            if ((i == 1 && rank_row == 0) || (i == m - 2 && rank_row == q - 1) ||
                (j == 1 && rank_col == 0) || (j == n - 2 && rank_col == q - 1)) {
                A[i * n + j] = A_prev[i * n + j];
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


int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Global configration
    int m = 2000;
    int n = 500;
    int p = size;
    int q = static_cast<int>(sqrt(p));

    // Entries of Matrix position
    int local_m = m / q;
    int local_n = n / q;
    int extra_m = m % q;
    int extra_n = n % q;

    // Processor position
    int row_rank = rank / q;
    int col_rank = rank % q;
    int start_row = row_rank * local_m + min(row_rank, extra_m);
    int start_col = col_rank * local_n + min(col_rank, extra_n);

    if (row_rank < extra_m) {
        local_m++;
    }
    if (col_rank < extra_n) {
        local_n++;
    }

    // Define ghost layers
    int extended_m = local_m + 2;  // Upper and Down
    int extended_n = local_n + 2;  // Left and Right

    double* A = new double[extended_m * extended_n];
    double* A_prev = new double[extended_m * extended_n];

    initialize_matrix(A, extended_m, extended_n, start_row, start_col, row_rank, col_rank, q);
    memcpy(A_prev, A, extended_m * extended_n * sizeof(double));

    // Synchronize Barrier--------------------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    double* buf_col = new double[local_m];
    double* buf_row = new double[local_n];

    for (int iter = 0; iter < ITERATIONS; iter++) {
        memcpy(A_prev, A, extended_m * extended_n * sizeof(double));

        // Neighber processor
        int up_rank = (row_rank > 0) ? rank - q : MPI_PROC_NULL;
        int down_rank = (row_rank < q - 1) ? rank + q : MPI_PROC_NULL;
        int left_rank = (col_rank > 0) ? rank - 1 : MPI_PROC_NULL;
        int right_rank = (col_rank < q - 1) ? rank + 1 : MPI_PROC_NULL;

        // Left - Right communication----------------------------------------------------------
        // Right edge
        if (col_rank != q - 1) {
            for (int i = 1; i < extended_m - 1; i++) {
                buf_col[i - 1] = A_prev[i * extended_n + local_n];
            }
            MPI_Send(buf_col, extended_m - 2, MPI_DOUBLE, right_rank, 0, MPI_COMM_WORLD);
        }
        
        // Recieve from right edge
        if (col_rank != 0) {
            MPI_Recv(buf_col, extended_m - 2, MPI_DOUBLE, left_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 1; i < extended_m - 1; i++) {
                A_prev[i * extended_n + 0] = buf_col[i - 1];
            }
        }
        // Left edge
        if (col_rank != 0) {
            for (int i = 1; i < extended_m - 1; i++) {
                buf_col[i - 1] = A_prev[i * extended_n + 1];
            }
            MPI_Send(buf_col, extended_m - 2, MPI_DOUBLE, left_rank, 1, MPI_COMM_WORLD);
        }

        // Recieve from left edge
        if (col_rank != q - 1) {
            MPI_Recv(buf_col, extended_m - 2, MPI_DOUBLE, right_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 1; i < extended_m - 1; ++i) {
                A_prev[i * extended_n + extended_n - 1] = buf_col[i - 1];
            }
        }

        // Upper - Bottom communication-----------------------------------------------------------

        // Upper edeg
        if (row_rank != 0) {
            memcpy(buf_row, &A_prev[1 * extended_n + 1], (extended_n - 2) * sizeof(double));
            MPI_Send(buf_row, extended_n - 2, MPI_DOUBLE, up_rank, 2, MPI_COMM_WORLD);
        }
        if (row_rank != q - 1) {
            MPI_Recv(buf_row, extended_n - 2, MPI_DOUBLE, down_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // Store in the bottom ghost edge
            memcpy(&A_prev[(extended_m - 1) * extended_n + 1], buf_row, (extended_n - 2) * sizeof(double));
        }

        // Bottom edeg
        if (row_rank != q - 1) {
            memcpy(buf_row, &A_prev[(extended_m - 2) * extended_n + 1], (extended_n - 2) * sizeof(double));
            MPI_Send(buf_row, extended_n - 2, MPI_DOUBLE, down_rank, 3, MPI_COMM_WORLD);
        }
        if (row_rank != 0) {
            MPI_Recv(buf_row, extended_n - 2, MPI_DOUBLE, up_rank, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // Store in the upper ghost edge
            memcpy(&A_prev[0 * extended_n + 1], buf_row, (extended_n - 2) * sizeof(double));
        }

        update_matrix(A, A_prev, extended_m, extended_n, row_rank, col_rank, q);
        // Synchronize Barrier--------------------------------------------------------------------
        MPI_Barrier(MPI_COMM_WORLD);
    }


    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;
    

    // Verification
    double local_sum = 0.0, local_sum_sqr = 0.0;
    for (int i = 1; i <= local_m; i++) {
        for (int j = 1; j <= local_n; j++) {
            double val = A[i * extended_n + j];
            local_sum += val;
            local_sum_sqr += pow(val, 2);
        }
    }

    double global_sum = 0.0, global_sum_sqr = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sum_sqr, &global_sum_sqr, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Print Result using root processor
    if (rank == 0) {
        cout << "Global sum: " << global_sum << endl;
        cout << "Global sum of squares: " << global_sum_sqr << endl;
        cout << "Elapsed time: " << elapsed_time << " seconds" << endl;
    }

    // Free buffer
    delete[] buf_col;
    delete[] buf_row;
    delete[] A;
    delete[] A_prev;
    MPI_Finalize();
    return 0;
}
