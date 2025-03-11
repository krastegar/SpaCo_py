// Title: Imhof (1961) algorithm
// Ref. (book or article): {J. P. Imhof, Computing the Distribution of Quadratic Forms in Normal Variables, Biometrika, Volume 48, Issue 3/4 (Dec., 1961), 419-426

// Description:
// Distribution function (survival function in fact) of quadratic forms in normal variables using Imhof's method.

#include <cmath>
#include <vector>
#include <functional>
#include <numeric>
#include <boost/math/quadrature/simpson.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

extern "C" {

    double theta(double *u, double *lambda, int *lambdalen, double *h, double *x, double *delta2) {
        int i, m;
        double sum = 0.0;
        m = lambdalen[0];
        for (i = 0; i <= m; i = i + 1) sum = sum + h[i] * std::atan(lambda[i] * u[0]) + delta2[i] * lambda[i] * u[0] / (1.0 + pow(lambda[i] * u[0], 2.0));
        sum = 0.5 * sum - 0.5 * x[0] * u[0];
        return(sum);
    }

    double rho(double *u, double *lambda, int *lambdalen, double *h, double *delta2) {
        int i, m;
        double prod = 1.0;
        m = lambdalen[0];
        for (i = 0; i <= m; i = i + 1)  prod = prod * pow(1.0 + pow(lambda[i] * u[0], 2.0), 0.25 * h[i]) * std::exp(0.5 * delta2[i] * pow(lambda[i] * u[0], 2.0) / (1.0 + pow(lambda[i] * u[0], 2.0)));
        return(prod);
    }

    double imhoffunc(double *u, double *lambda, int *lambdalen, double *h, double *x, double *delta2) {
        double theta(double *u, double *lambda, int *lambdalen, double *h, double *x, double *delta2);
        double rho(double *u, double *lambda, int *lambdalen, double *h, double *delta2);
        double res;
        res = (std::sin(theta(u, lambda, lambdalen, h, x, delta2))) / (u[0] * rho(u, lambda, lambdalen, h, delta2));
        return(res);
    }

    typedef void integr_fn(double *x, int n, void *ex);

    void f(double *x, int n, void *ex) {
        double imhoffunc(double *u, double *lambda, int *lambdalen, double *h, double *x, double *delta2);
        int i;
        double *xx;
        xx = new double[1];
        xx[0] = ((double*)ex)[0];
        int *lambdalen;
        lambdalen = new int[1];
        lambdalen[0] = (int)(((double*)ex)[1]);
        double *lambda;
        lambda = new double[lambdalen[0]];
        for (i = 1; i <= lambdalen[0]; i = i + 1) lambda[i - 1] = ((double*)ex)[i + 1];
        double *h;
        h = new double[lambdalen[0]];
        for (i = 1; i <= lambdalen[0]; i = i + 1) h[i - 1] = ((double*)ex)[lambdalen[0] + i + 1];
        double *delta2;
        delta2 = new double[lambdalen[0]];
        for (i = 1; i <= lambdalen[0]; i = i + 1) delta2[i - 1] = ((double*)ex)[2 * lambdalen[0] + i + 1];
        double *u;
        u = new double[1];
        for (i = 1; i <= n; i = i + 1) {
            u[0] = x[i - 1];
            x[i - 1] =  imhoffunc(u, lambda, lambdalen, h, xx, delta2);
        }

        delete[] xx;
        delete[] lambdalen;
        delete[] lambda;
        delete[] h;
        delete[] delta2;
        delete[] u;
    }
    double integrate_function(std::function<double(double)> f, double lower_bound, double upper_bound, double epsabs, double epsrel) {
        using namespace boost::math::quadrature;
        double result = simpson(f, lower_bound, upper_bound, epsabs, epsrel);
        return result;
    }
    void probQsupx(double *x, double *lambda, int *lambdalen, double *h, double *delta2, double *Qx, double *epsabs, double *epsrel, int *limit) {
        int i;
        void f(double *x, int n, void *ex);

        double *ex;
        ex = new double[2 + 3 * lambdalen[0]];
        ex[0] = x[0];
        ex[1] = (double)lambdalen[0];
        for (i = 1; i <= lambdalen[0]; i = i + 1) ex[i + 1] = lambda[i - 1];
        for (i = 1; i <= lambdalen[0]; i = i + 1) ex[lambdalen[0] + i + 1] = h[i - 1];
        for (i = 1; i <= lambdalen[0]; i = i + 1) ex[2 * lambdalen[0] + i + 1] = delta2[i - 1];

        double *result;
        result = new double[1];
        double *abserr;
        abserr = new double[1];
        double lower_bound = 0.0;
        double upper_bound = std::numeric_limits<double>::infinity();
        integrate_function(f, lower_bound, upper_bound, *epsabs, *epsrel);
        Qx[0] = 0.5 + result[0] / M_PI;

        epsabs[0] = abserr[0];

        delete[] ex;
        delete[] result;
        delete[] abserr;

        return;
    }
}

PYBIND11_MODULE(imhoff, m) {
    m.def("theta", &theta, "Calculate theta",
          py::arg("u"), py::arg("lambda"), py::arg("lambdalen"), py::arg("h"), py::arg("x"), py::arg("delta2"));
    m.def("rho", &rho, "Calculate rho",
          py::arg("u"), py::arg("lambda"), py::arg("lambdalen"), py::arg("h"), py::arg("delta2"));
    m.def("imhoffunc", &imhoffunc, "Calculate imhoffunc",
          py::arg("u"), py::arg("lambda"), py::arg("lambdalen"), py::arg("h"), py::arg("x"), py::arg("delta2"));
    m.def("probQsupx", &probQsupx, "Calculate probQsupx",
          py::arg("x"), py::arg("lambda"), py::arg("lambdalen"), py::arg("h"), py::arg("delta2"),
          py::arg("Qx"), py::arg("epsabs"), py::arg("epsrel"), py::arg("limit"));
}