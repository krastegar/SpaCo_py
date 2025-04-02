#include <cmath>
#include <vector>
#include <functional>
#include <numeric>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/chrono.h>
#include <pybind11/functional.h>
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
        double *ex_data = static_cast<double*>(ex);
        int lambdalen = static_cast<int>(ex_data[1]);

        // Use stack-allocated vectors for lambda, h, and delta2
        std::vector<double> lambda(ex_data + 2, ex_data + 2 + lambdalen);
        std::vector<double> h(ex_data + 2 + lambdalen, ex_data + 2 + 2 * lambdalen);
        std::vector<double> delta2(ex_data + 2 + 2 * lambdalen, ex_data + 2 + 3 * lambdalen);

        double xx = ex_data[0];
        double u;

        for (int i = 0; i < n; i++) {
            u = x[i];
            x[i] = imhoffunc(&u, lambda.data(), &lambdalen, h.data(), &xx, delta2.data());
        }
    }

    double integrate_function(std::function<double(double)> f, double lower_bound, double upper_bound, double epsabs, double epsrel) {
        using namespace boost::math::quadrature;
        gauss_kronrod<double, 15> integrator;
        double result = integrator.integrate(f, lower_bound, upper_bound, epsabs, epsrel);
        return result;
    }

    double probQsupx(double x, const std::vector<double>& lambda, int lambdalen, const std::vector<double>& h, const std::vector<double>& delta2, double epsabs, double epsrel, int limit) {
        // Allocate memory on the stack using std::vector
        std::vector<double> ex(2 + 3 * lambdalen);

        // Populate the ex array
        ex[0] = x;
        ex[1] = static_cast<double>(lambdalen);
        for (int i = 0; i < lambdalen; i++) ex[i + 2] = lambda[i];
        for (int i = 0; i < lambdalen; i++) ex[lambdalen + i + 2] = h[i];
        for (int i = 0; i < lambdalen; i++) ex[2 * lambdalen + i + 2] = delta2[i];

        double lower_bound = 0.0;
        double upper_bound = std::numeric_limits<double>::infinity();

        // Using a lambda function to wrap the f function to pass it to the integrator
        auto f_wrapper = [&](double x) {
            double result;
            f(&x, 1, ex.data()); // Pass the raw pointer to the std::vector data
            result = x;
            return result;
        };

        // Perform the integration
        double result = integrate_function(f_wrapper, lower_bound, upper_bound, epsabs, epsrel);

        // Compute Qx
        double Qx = 0.5 + result / M_PI; // M_PI is defined in cmath as pi

        return Qx;
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
          py::arg("epsabs"), py::arg("epsrel"), py::arg("limit"));
}
