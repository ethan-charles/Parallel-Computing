// Yichen Lu nechy@umich.edu
#include "f.h"
#include <stdio.h>
#include <omp.h>
#include <chrono>
#include <iostream>
#include <limits>
#include <queue>

using std::cout;
using std::endl;
using std::max;
using std::pair;
using std::queue;
using std::vector;

int main(int argc, char** argv) {
    int p = atoi(argv[1]);
    double final_res = max(f(a), f(b));
    queue<pair<double, double>> tasks;
    vector<bool> status(p, false);

    tasks.push({a, (a + b) / 2});
    tasks.push({(a + b) / 2, b});

    omp_lock_t tasks_lock;
    omp_lock_t status_lock;
    omp_init_lock(&tasks_lock);
    omp_init_lock(&status_lock);

    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel num_threads(p) shared(tasks, status, final_res)
    {
        int current_id = omp_get_thread_num();
        double M = max(f(a), f(b));
        
        while (1) {
            pair<double, double> current_task = {0.0, 0.0};
            bool all_done = true;
            bool has_task = false;
            omp_set_lock(&tasks_lock);
            if (!tasks.empty()) {
                current_task = tasks.front();
                tasks.pop();
                status[current_id] = true;
                all_done = false;
                has_task = true;
            }
            omp_unset_lock(&tasks_lock);

            if (!has_task && all_done) { 
                break; 
            }

            if (!has_task) {
                continue;
            }

            double lower = current_task.first;
            double upper = current_task.second;
            double most = (f(lower) + f(upper) + s * (upper - lower)) / 2;
            M = max(M, max(f(lower), f(upper)));

            if (most >= M + epsilon) {
                double new_l_lower = lower;
                double new_l_upper = (lower + upper) / 2;
                double new_r_lower = (lower + upper) / 2;
                double new_r_upper = upper;

                omp_set_lock(&tasks_lock);
                tasks.push({new_l_lower, new_l_upper});
                tasks.push({new_r_lower, new_r_upper});
                omp_unset_lock(&tasks_lock);
            }

            #pragma omp critical
            {
                final_res = max(final_res, M);
            }

            omp_set_lock(&status_lock);
            status[current_id] = false;
            omp_unset_lock(&status_lock);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;

    cout << "Execution Time: " << duration.count() << " s" << endl;
    cout << "Max value: " << final_res << endl;

    omp_destroy_lock(&tasks_lock);
    omp_destroy_lock(&status_lock);

    return;
}
