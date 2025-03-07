#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>
#include <functional>
#include <numeric>

namespace py = pybind11;

// Compute theta function
double theta(const std::vector<double>& u, const std::vector<double>& lambda,
             const std::vector<double>& h, const std::vector<double>& x, const std::vector<double>& delta2) {
    int m = lambda.size();
    double sum = 0.0;
    for (int i = 0; i < m; i++) {
        sum += h[i] * std::atan(lambda[i] * u[0]) + delta2[i] * lambda[i] * u[0] / (1.0 + std::pow(lambda[i] * u[0], 2.0));
    }
    return 0.5 * sum - 0.5 * x[0] * u[0];
}

// Compute rho function
double rho(const std::vector<double>& u, const std::vector<double>& lambda,
           const std::vector<double>& h, const std::vector<double>& delta2) {
    int m = lambda.size();
    double prod = 1.0;
    for (int i = 0; i < m; i++) {
        prod *= std::pow(1.0 + std::pow(lambda[i] * u[0], 2.0), 0.25 * h[i]) *
                std::exp(0.5 * delta2[i] * std::pow(lambda[i] * u[0], 2.0) / (1.0 + std::pow(lambda[i] * u[0], 2.0)));
    }
    return prod;
}

// Function under the integral sign in Imhof's equation
double imhoffunc(const std::vector<double>& u, const std::vector<double>& lambda,
                 const std::vector<double>& h, const std::vector<double>& x, const std::vector<double>& delta2) {
    return std::sin(theta(u, lambda, h, x, delta2)) / (u[0] * rho(u, lambda, h, delta2));
}

// Numerical integration function using the trapezoidal rule
double integrate(std::function<double(double)> func, double a, double b, int n = 1000) {
    double h = (b - a) / n;
    double sum = 0.5 * (func(a) + func(b));
    for (int i = 1; i < n; ++i) {
        sum += func(a + i * h);
    }
    return sum * h;
}

// Compute probability Qsupx
double probQsupx(double x, const std::vector<double>& lambda, const std::vector<double>& h,
                 const std::vector<double>& delta2, double epsabs = 1e-6, double epsrel = 1e-6, int limit = 10000) {
    auto integral_func = [&](double u) {
        return imhoffunc({u}, lambda, h, {x}, delta2);
    };
    double result = integrate(integral_func, 0.0, 10.0, limit); // Adjust bounds as needed
    return 0.5 + result / M_PI;
}

PYBIND11_MODULE(imhoff, m) {
    m.def("imhoffunc", &imhoffunc, "Compute the Imhof function",
          py::arg("u"), py::arg("lambda"), py::arg("h"), py::arg("x"), py::arg("delta2"));
    m.def("probQsupx", &probQsupx, "Compute probability Qsupx",
          py::arg("x"), py::arg("lambda"), py::arg("h"), py::arg("delta2"),
          py::arg("epsabs") = 1e-6, py::arg("epsrel") = 1e-6, py::arg("limit") = 10000);
}
