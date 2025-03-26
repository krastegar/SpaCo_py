import numpy as np
import scipy.integrate as spi
from typing import List, Callable


def theta(
    u: List[float],
    lambda_: List[float],
    h: List[float],
    x: List[float],
    delta2: List[float],
) -> float:
    """
    Compute the theta function for Imhof's equation.

    Parameters:
    u (List[float]): Integration variable (typically a single-element list)
    lambda_ (List[float]): Eigenvalues of the covariance matrix
    h (List[float]): Coefficients representing weights
    x (List[float]): Upper quantiles for probability computation
    delta2 (List[float]): Delta squared values for computation

    Returns:
    float: Computed theta value
    """
    m = len(lambda_)  # Number of elements in lambda_
    sum_ = sum(
        h[i] * np.arctan(lambda_[i] * u[0])
        + delta2[i] * lambda_[i] * u[0] / (1.0 + (lambda_[i] * u[0]) ** 2)
        for i in range(m)
    )
    return 0.5 * sum_ - 0.5 * x[0] * u[0]  # Final theta computation


def rho(
    u: List[float], lambda_: List[float], h: List[float], delta2: List[float]
) -> float:
    """
    Compute the rho function for Imhof's equation.

    Parameters:
    u (List[float]): Integration variable (typically a single-element list)
    lambda_ (List[float]): Eigenvalues of the covariance matrix
    h (List[float]): Coefficients representing weights
    delta2 (List[float]): Delta squared values for computation

    Returns:
    float: Computed rho value
    """
    m = len(lambda_)  # Number of elements in lambda_
    prod = np.prod(
        [
            (1.0 + (lambda_[i] * u[0]) ** 2) ** (0.25 * h[i])
            * np.exp(
                0.5
                * delta2[i]
                * (lambda_[i] * u[0]) ** 2
                / (1.0 + (lambda_[i] * u[0]) ** 2)
            )
            for i in range(m)
        ]
    )
    return prod  # Final rho computation


def imhoffunc(
    u: List[float],
    lambda_: List[float],
    h: List[float],
    x: List[float],
    delta2: List[float],
) -> float:
    """
    Compute the function under the integral sign in Imhof's equation.

    Parameters:
    u (List[float]): Integration variable (typically a single-element list)
    lambda_ (List[float]): Eigenvalues of the covariance matrix
    h (List[float]): Coefficients representing weights
    x (List[float]): Upper quantiles for probability computation
    delta2 (List[float]): Delta squared values for computation

    Returns:
    float: Computed Imhof function value
    """
    return np.sin(theta(u, lambda_, h, x, delta2)) / (u[0] * rho(u, lambda_, h, delta2))


def integrate(
    func: Callable[[float], float], a: float, b: float, n: int = 1000
) -> float:
    """
    Perform numerical integration using the trapezoidal rule.

    Parameters:
    func (Callable[[float], float]): Function to integrate
    a (float): Lower integration bound
    b (float): Upper integration bound
    n (int): Number of subdivisions (default: 1000)

    Returns:
    float: Approximate integral value
    """
    x = np.linspace(a, b, n + 1)  # Create equally spaced points
    y = np.array([func(xi) for xi in x])  # Evaluate function at each point
    return np.trapezoid(y, x)  # Compute integral using the trapezoidal rule


def probQsupx(
    x: float,
    lambda_: List[float],
    h: List[float],
    delta2: List[float],
    epsabs: float = 1e-6,
    epsrel: float = 1e-6,
    limit: int = 10000,
) -> float:
    """
    Compute probability Qsupx using numerical integration.

    Parameters:
    x (float): Upper quantile value
    lambda_ (List[float]): Eigenvalues of the covariance matrix
    h (List[float]): Coefficients representing weights
    delta2 (List[float]): Delta squared values for computation
    epsabs (float): Absolute integration tolerance (default: 1e-6)
    epsrel (float): Relative integration tolerance (default: 1e-6)
    limit (int): Maximum number of function evaluations (default: 10000)

    Returns:
    float: Computed probability value
    """

    def integral_func():
        lambda u: imhoffunc(
            [u], lambda_, h, [x], delta2
        )  # Wrapper function for integration

    result, _ = spi.quad(
        integral_func, 0.0, 10.0, limit=limit, epsabs=epsabs, epsrel=epsrel
    )  # Perform integration
    return 0.5 + result / np.pi  # Compute final probability value
