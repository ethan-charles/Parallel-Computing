#ifndef F_H
#define F_H

#include <cmath>

double a = 1.0;
double b = 100.0;
double epsilon = 1e-6;
double s = 12.0;

double f(double x) {
    double ans = 0;
    for (int i = 100; i >= 1; --i) {
        double temp = 0;
        for (int j = i; j >= 1; --j) temp += pow(x + 0.5 * j, -3.3);
        ans += sin(x + temp) / pow(1.3, i);
    }
    return ans;
}


#endif