# ============================================================================
# SPACO Class Overview
# ============================================================================
#
# Description:
# -----------
# The SPACO class is designed for Spectral Filtering and Projection using
# Principal Components Analysis (PCA) and Graph Laplacian. It takes in sample
# features and a neighbor matrix as inputs and provides methods for
# preprocessing, PCA whitening, spectral filtering, and projection.

# Methods
# -------
#
# The following methods are available in the SPACO class:
#
# 1. `__init__`: Initializes a SPACO object with sample features, neighbor matrix,
#    and optional parameters for PCA and spectral filtering.
#
# 2. `preprocess`: Preprocesses the sample features array by ensuring it is in
#    the correct shape, removing constant features, and scaling using StandardScaler.
#
# 3. `pca_whitening`: Performs PCA whitening on the sample features array and
#    returns the whitened data. Optional parameters include the threshold (c) for
#    selecting the variance of the principal components
#
# 4. `_resample_lambda_cut`: Resamples the eigenvalues of the graph Laplacian
#    matrix to estimate the eigenvalue threshold for spectral filtering. Uses
#    percentile and resample_iterations as inputs.
#
# 5. `spectral_filtering`: Performs spectral filtering on the whitened data using
#    the estimated eigenvalue threshold and returns the filtered eigenvalues and
#    eigenvectors.
#
# 6. `spaco_projection`: Projects the original data onto the filtered eigenvectors
#    and returns the projected data.
#
# 7. `spaco_test`: Computes a test statistic for a given input vector x using the
#    projected data and graph Laplacian.
#
# ============================================================================
# Date: 20/02/2025
# Author: Kiarash Rastegar
# ============================================================================

import numpy as np
import imhoff 
from scipy.linalg import eigh
from scipy.sparse.linalg import eigs
from scipy.stats import t
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import StandardScaler

"""
Next step is to find a way to not rerun the resampling method more than once across
all other different methods. I need to find a way for the results to be stored. I really wonder 
which test are running the method multiple times. I think it is the spaco_test method.
but not sure.......currently for small dataset it takes 37mins to run whole pipeline.
I think the issue is that the spaco_test method is called multiple times for each gene in the dataset.


Next plan of attack: (26/04/2025)
1. figure out how to run each method only once
2. figure out why the matrix Vk does not have expected dimensions (n,k), most likely related to the resampling method. 
"""
class SPACO:
    def __init__(
        self, sample_features, neighbormatrix, c=0.95, compute_nSpacs=True, percentile=95
    ):
        """
        Initialize a SpaCo object.

        Parameters
        ----------
        sample_features : array-like (n_samples, n_features)
            The sample features array.
        neighbormatrix : array-like (n_samples, n_samples)
            The neighbor matrix.
        c : float, optional
            The threshold for selecting the minimal number of principal components. Default is 0.95.
        lambda_cut : float, optional
            The eigenvalue threshold for selecting the minimal number of principal components. Default is None.
        percentile : int, optional
            The percentile of the eigenvalues that is used to set the eigenvalue threshold. Default is 95.

        Returns
        -------
        None
        """
        # initial variables / attributes
        self.percentile: int = percentile
        self.compute_nSpacs: bool = compute_nSpacs
        self.SF: np.ndarray = self.__preprocess(sample_features)
        self.A: np.ndarray = self.__check_if_square(neighbormatrix)
        self.c: float = c

        # Additional attributes to store intermediate results:
        self.whitened_data: np.ndarray = self.__pca_whitening()
        
        # results of the spectral filtering (self.sampled_sorted_eigvecs, self.sampled_sorted_eigvals, self.graphLaplacian)
        self.lambda_cut: float = None
        self.spectral_results: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] = self.__spectral_filtering()
        self.graphLaplacian: np.ndarray = self.spectral_results[2]
        self.sampled_sorted_eigvecs: np.ndarray = self.spectral_results[0]
        self.sampled_sorted_eigvals: np.ndarray = self.spectral_results[1]

        # results of the projection
        self.Pspac: np.ndarray
        self.Vk: np.ndarray
        self.Pspac, self.Vk = self.spaco_projection()

        # results of the test
        self.sigma: np.ndarray
        self.sigma_eigh: np.ndarray
        self.sigma_eigh, self.sigma = self.__sigma_eigenvalues()

    def __remove_constant_features(self, X: np.ndarray) -> np.ndarray:
        """
        Remove constant features from the data using variance.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        X : array-like, shape (n_samples, n_features)
            The input data with constant features removed.

        Raises
        ------
        ValueError
            If all features are constant, a ValueError is raised.
        """
        # Remove constant features
        X = X[:, np.var(X, axis=0) > 0]

        if np.all(X == 0):
            raise ValueError("No features left after removing constant features.")
        return X

    def __preprocess(self, X: np.ndarray) -> np.ndarray:
        """
        Check to see if data type is numpy array and preprocess the sample features array.

        The preprocessing steps are:
        1. Ensure X is spots × features
        2. Remove constant features
        3. Scale the features using StandardScaler

        Args:
            sample_features: Spots × Features array to be preprocessed

        Returns:
            Preprocessed Spots × Features array
        """
        # Check if the input is a numpy array
        if not isinstance(X, np.ndarray):
            raise ValueError(
                """
                Input must be a numpy array (np.ndarray). Input is most likely a pandas dataframe (pd.DataFrame). 
                Please convert to numpy array using to_numpy() if this is the case.
                """
            )

        # Initialize StandardScaler
        scaler = StandardScaler()

        # checking to see if the input is a vector
        X = self.__remove_constant_features(X)

        # returning scaled and centered features (using z-scaling)
        return scaler.fit_transform(X)

    def __check_if_square(self, X: np.ndarray) -> bool:
        """
        Check if the neighborhood matrix is a square matrix and is numpy array.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        """
        if not isinstance(X, np.ndarray):
            raise ValueError(
                """
                Input must be a numpy array (np.ndarray). Input is most likely a pandas dataframe (pd.DataFrame). 
                Please convert to numpy array using to_numpy() if this is the case.
                """
            )
        # Check if the input is a square matrix
        if X.shape[0] != X.shape[1]:
            raise ValueError(
                "The input data is not a square matrix. Please provide a square matrix."
            )
        return X

    def __orthogonalize(
        self,
        X: np.ndarray,
        A: np.ndarray,
        nSpacs: int,
        tol: float = np.sqrt(np.finfo(float).eps),
    ) -> np.ndarray:
        """
        Version of QR factorization that Achim, David and Niklaus came up with
        for the SPACO algorithm. Want to talk to achim about replacing this whole thing
        just with np.linalg.qr() function. This implents a gram-schmidt orthogonalization
     self   of the columns of the projection matrix X and returns a Q.

        Parameters
        ----------
        X : np.ndarray
            The projection matrix.
        A : np.ndarray
            The neighbor matrix.
        nSpacs : int
            The number of SpaCo components to retain.

        Returns
        -------
        np.ndarray
            The orthogonalized (unitary) matrix.
        """

        # preFactor = 1 # not sure what this is for

        # getting number of rows and columns of the projection matrix X
        m: int = X.shape[0]
        n: int = X.shape[1]
        if m < nSpacs:
            raise ValueError(
                "The number of rows of the projection matrix must be greater than or equal to the number of columns."
            )
        Q: np.ndarray = np.zeros(
            (m, n)
        )  # initializing the orthogonalized (unitary) matrix
        norms: np.ndarray = np.zeros(
            n
        )  # initializing the norms of the columns of the projection matrix (this is a vector)
        for k in range(nSpacs):
            Q[:, k] = X[
                :, k
            ]  # setting the k-th column of the orthogonalized matrix to the k-th column of the projection matrix
            if k > 0:
                for i in range(k):
                    Q[:, k] -= (
                        (Q[:, k] @ A @ Q[:, i]) / (Q[:, i] @ A @ Q[:, i]) * Q[:, i]
                    )
            scalar_product = Q[:, k] @ A @ Q[:, k]
            if scalar_product < 0:
                raise ValueError(
                    f"Scalar product is negative: {scalar_product} for kth column {k}"
                )
            norms[k] = np.sqrt(scalar_product)
            if abs(norms[k]) < tol:
                raise ValueError("MATRIX [A] IS NOT FULL RANK.")
            Q[:, k] /= norms[k]  # not sure why in R version its c(Norms[k])
        return Q

    def __pca_whitening(self, c=0.95):
        """
        Perform PCA whitening on the input data X.

        Parameters:
        X (numpy array): Input data, shape (n_samples, n_features)
        c (float): Threshold for selecting the minimal number of principal components. Default is 0.95.

        Returns:
        X_whitened (numpy array): Whitened data, shape (n_samples, n_features)
        """
        # Step 1: Center the data
        # Subtract the mean of each feature from the dataset, so the dataset has zero mean.
        # Centering the data means subtracting the mean of the data from
        # each data point. This is important because whitening is a
        # linear transformation that depends on the mean of the data.

        # Step 2: Compute the covariance matrix
        # The covariance matrix is a square, symmetric matrix where the
        # element at row i and column j is the covariance of the i-th and
        # j-th features of the dataset.
        # The covariance matrix is a measure of how much the data varies in
        # each direction. It's a measure of how spread out the data is.
        cov = np.cov(self.SF, rowvar=False)
        # print(f"Dimensions of covariance matrix: {cov.shape}")

        # Step 3: Compute the eigenvectors and eigenvalues of the covariance matrix
        # Eigenvalues are scalars and eigenvectors are vectors. Eigenvectors
        # are the directions in which the data varies the most, and
        # eigenvalues are the amount of variation in those directions.
        # The eigenvectors are the principal components of the data.
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Step 4: Sort the eigenvectors by eigenvalue in descending order
        # The eigenvectors are sorted in descending order of their
        # corresponding eigenvalues. This is because the largest
        # eigenvalue/eigenvector pair captures the most variance in the
        # data.
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Step 5: Select minimal number of components such that the variance threshold c is satisfied
        total_variance = np.sum(eigenvalues)
        cumulative_variance = np.cumsum(eigenvalues)
        r = np.searchsorted(cumulative_variance / total_variance, c) + 1

        # Step 6: Select top r eigenvalues and eigenvectors
        W_r = eigenvectors[:, :r]
        D_r = np.diag(eigenvalues[:r])
        D_r_inv_sqrt = np.linalg.inv(np.sqrt(D_r))

        # Step 7: Compute whitened data
        self.whitened_data = np.dot(self.SF, W_r).dot(D_r_inv_sqrt)
        # print(f"Dimensions of whitened data: {X_whitened.shape}")

        if self.whitened_data.shape[0] != self.SF.shape[0]:
            raise ValueError(f"Whitened Data has wrong dimensions: {self.whitened_data.shape}")
        return self.whitened_data

    def __shuffle_decomp(self) -> float:
        """
        Shuffles the rows of the input matrix X and computes the largest eigenvalue
        of the shuffled matrix using the eigs function.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        largest_eigenvalue : float
            The largest eigenvalue of the shuffled matrix.
        """
        # This is done to break any structure in the data to represent randomness
        # We shuffle the rows of the input matrix X
        X_shuffled = self.A[np.random.permutation(self.A.shape[0]), :]
        
        # The graph Laplacian is more accurately a kernel constructed by using the spatial information in the 
        # neighbor matrix. The graph Laplacian is a symmetric matrix.
        # We first create a matrix of ones
        L: np.ndarray = (1 / X_shuffled.shape[0]) * np.eye(X_shuffled.shape[0]) + (
            1 / np.abs(X_shuffled).sum()
        ) * X_shuffled
        
        # Compute the matrix M which is the product of the whitened data and the graph Laplacian
        # M is a symmetric matrix
        M = self.whitened_data.T @ L @ self.whitened_data
        
        # Compute the largest eigenvalue of M
        
        # eigs returns the eigenvalues and eigenvectors of M
        # We only need the largest eigenvalue so we set k=1
        # The eigenvectors are not needed so we set which="LR"
        largest_eigenvalue = eigs(M, k=1, which="LR", maxiter=1000, tol=1e-4)[0][0].real

        return largest_eigenvalue

    def __CI_SE(self, results_all: list[float]) -> Tuple[float, float]:
        """
        Compute the 95% confidence interval and standard error of the mean for a list of values.

        Parameters
        ----------
        results_all : List[float]
            A list of values to compute the confidence interval and standard error for.

        Returns
        -------
        ci_lower : float
            The lower bound of the 95% confidence interval.
        ci_upper : float
            The upper bound of the 95% confidence interval.
        """
        # Calculate the mean of the results
        mean: float = np.mean(results_all)

        # Calculate the standard error of the mean
        # np.std calculates the standard deviation of the list
        # Divide by the square root of the number of observations
        std_error: float = np.std(results_all) / np.sqrt(len(results_all))

        # Determine the t critical value for 95% confidence
        # t.ppf gives the value of the t-distribution for a given cumulative probability
        # 0.975 is used to find the two-tailed critical value for 95% confidence
        # df is degrees of freedom, which is number of observations minus one
        t_critical: float = t.ppf(0.975, df=len(results_all) - 1)

        # Calculate the margin of error
        # This is the product of the t critical value and the standard error
        margin_of_error: float = t_critical * std_error

        # Calculate the lower and upper bounds of the confidence interval
        ci_lower: float = mean - margin_of_error
        ci_upper: float = mean + margin_of_error

        return ci_lower, ci_upper

    def replicate(self, n_iterations: int) -> list[float]:
        """
        Replicates the __shuffle_decomp method n_iterations times.

        Parameters
        ----------
        n_iterations : int
            The number of times to replicate the __shuffle_decomp method.

        Returns
        -------
        results_all : list[float]
            A list of the results of each replication.
        """
        with ThreadPoolExecutor() as executor:
            results_all = list(executor.map(lambda _: self.__shuffle_decomp(), range(n_iterations)))
        return results_all

    def __resample_lambda_cut(
        self,
        batch_size: int = 10,
        n_iterations: int = 100,
        n_simulations: int = 1000,
    ) -> float:
        
        """
        Resamples the shuffled adjacency matrix to calculate the confidence interval for the relevant number of SpaCs.
        Generating a confidence interval from the shuffled M matrix eigenvalues representing random noise. The 
        CI is then used to determine the relevant number of SpaCs.
        The method iteratively decreases the confidence interval until the number of eigenvalues within the
        confidence interval is 1 or the number of iterations exceeds n_simulations.

        Parameters
        ----------
        batch_size : int, optional (default=10)
            The number of times to replicate the __shuffle_decomp method in each iteration.
        n_iterations : int, optional (default=100)
            The number of times to replicate the __shuffle_decomp method.
        n_simulations : int, optional (default=1000)
            The maximum number of iterations to perform.

        Returns
        -------
        rel_spacs_idx : int
            The relevant number of SpaCs.
        """
        # shuffle / permute the neighbor matrix
        results_all: list = self.replicate(n_iterations)
        results_all: np.array = np.array(results_all)
        # calculate the 95 CI and SE
        ci_lower, ci_upper = self.__CI_SE(results_all)

        # Select the eigenvalues from results_all that are within the 95% CI
        # (i.e. the eigenvalues that are not significantly different from the null hypothesis)
        lambdas_inCI: np.ndarray = results_all[
            (results_all >= ci_lower) & (results_all <= ci_upper)
        ]

        iterations: int = 0
        # iteratively decreasing the CI interval until the lambdas_inCI is 1
        # or the number of iterations is greater than n_simulations
        print('Initializing resampling method to compute the number of relevant spatial components.....')

        while len(lambdas_inCI) > 1:
            iterations += 1
            # Adding the batch size to the number of iterations to steadily decrease the CI margins
            batch_results = self.replicate(batch_size)
            
            # Calculate the 95% CI and SE for the new batch of results
            np.append(results_all, batch_results)
            ci_lower, ci_upper = self.__CI_SE(results_all)

            # checking to see how many lambdas are in the CI
            lambdas_inCI = results_all[(results_all >= ci_lower) & (results_all <= ci_upper)]

            if len(lambdas_inCI) < 2:
                # lambdas_inCI should be a 1D array with only one element, which is the lambda of interest
                self.lambda_cut = lambdas_inCI[0]
                print(f'number of iterations: {iterations}.\n')
                return self.lambda_cut

            if iterations >= n_simulations:
                print(
                    f"Reached maximum number of iterations: {n_simulations}.\n # of elements in CI: {len(lambdas_inCI)}"
                )
                # if ther are still more than 1 eigenvalue in the CI, we take the upper bound of the CI
                self.lambda_cut = ci_upper
                return self.lambda_cut
  
    def __spectral_filtering(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform spectral filtering on the whitened data using the graph Laplacian.

        This method applies spectral filtering by computing the graph Laplacian
        and projecting the whitened data onto the eigenvectors of the resulting
        matrix. It uses a heuristic method to determine the eigenvalue threshold
        for filtering.

        Returns
        -------
        sampled_sorted_eigvecs : numpy.ndarray
            The eigenvectors corresponding to the largest eigenvalues after filtering.
        sampled_sorted_eigvals : numpy.ndarray
            The largest eigenvalues after filtering.
        whitened_da ta : numpy.ndarray
            The whitened data.
        L : numpy.ndarray
            The computed graph Laplacian.
        """
        # Declaring variables

        neighbor_matrix: np.ndarray = self.A
        eigvals: np.ndarray
        eigvecs: np.ndarray

        # Calculating Graph Laplacian and M matrix
        n: int = neighbor_matrix.shape[0]
        self.graphLaplacian: np.ndarray = (1 / n) * np.eye(n) + (
                    1 / np.abs(neighbor_matrix).sum()
                ) * neighbor_matrix
        
        # condition to determine if the graph laplacian was calculated correctly
        if neighbor_matrix.shape != self.graphLaplacian.shape:
            raise ValueError("Graph Laplacian has incorrect shape")

        M: np.ndarray = self.whitened_data.T @ self.graphLaplacian @ self.whitened_data

        # Check to see if M has the correct dimensions
        if M.shape[0] != M.shape[1]:
            raise ValueError("M matrix is not square issue with matrix multiplication")

        # Eigenvalue decomposition
        eigvals, eigvecs = eigh(M)
        idx: np.ndarray = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

        # Simple heuristic method for lambda cut (for now just median...later resampling)
        if self.compute_nSpacs is False:
            self.lambda_cut: int = np.median(eigvals)
            print(
                f"Using median eigenvalue as lambda cut: {self.lambda_cut}.\n")
        else: 
            self.lambda_cut: int = self.__resample_lambda_cut(
                batch_size=10,
                n_iterations=100,
                n_simulations=1000,
            )
        print(
            f"Using resampling method to compute the lambda cut: {self.lambda_cut}.\n"
        )
        # Filter the eigenvalues and eigenvectors based on the lambda cut
        k: int = len(eigvals[eigvals >= self.lambda_cut])  # index where lambda cut is
        self.sampled_sorted_eigvecs: np.ndarray = eigvecs[:, :k]
        self.sampled_sorted_eigvals: np.ndarray = eigvals[:k]

        return (self.sampled_sorted_eigvecs, self.sampled_sorted_eigvals, self.graphLaplacian)

    def spaco_projection(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project the sample features onto the SPACO space.

        This method projects the sample features onto the SPACO space by first
        performing spectral filtering and then computing the orthonormal
        matrix in the SPACO space.

        Returns
        -------
        projected_data: np.ndarray
            The projected sample features in the SPACO space.
        """
        # Declaring variables
        sample_feature_matrix: np.ndarray = self.SF
        
        # Generating orthonormal matrix in SPACO space
        U: np.ndarray = self.sampled_sorted_eigvecs / np.sqrt(self.sampled_sorted_eigvals)
        # print(f'First few vecs of view: {U[:, :5]}')
        self.Vk: np.ndarray = self.whitened_data @ U
        self.Pspac: np.ndarray = self.Vk @ self.Vk.T @ self.graphLaplacian @ sample_feature_matrix
        return self.Pspac, self.Vk

    def __sigma_eigenvalues(self) -> np.ndarray:
        """
        Compute the eigenvalues of the transformed matrix L @ Sk @ Sk.T @ L.

        This method computes the eigenvalues of the transformed matrix L @ Sk @ Sk.T @ L,
        where Sk is the matrix composed of the first nSpacs - 1 columns of the
        projection matrix Pspac. The eigenvalues are used as the coefficients
        to compute the test statistic.

        Returns
        -------
        sigma: np.ndarray
            The eigenvalues of the transformed matrix.
        """


        # Compute the number of SPACO components
        
        # Compute the matrix Sk by taking the first nSpacs - 1 columns of the projection matrix Pspac
        projection: np.ndarray = self.__orthogonalize(X=self.Vk, A=self.graphLaplacian, nSpacs=self.Vk.shape[1])
        Sk: np.ndarray = projection[
            :, :self.Vk.shape[1]
        ]  # in the R code, it is S = projection[, 1:nSpacs]

        # Compute the transformed matrix L @ Sk @ Sk.T @ L
        sigma: np.ndarray = self.graphLaplacian @ Sk @ Sk.T @ self.graphLaplacian # Sk.T @ self.graphLaplacian @ self.graphLaplacian @ Sk

        # Compute the eigenvalues of the sigma matrix
        sigma_eigh: np.ndarray = np.linalg.eigvalsh(sigma)

        return sigma_eigh, sigma

    def __psum_chisq(
        self, q, eig_vals, epsabs=float(10 ^ (-6)), epsrel=float(10 ^ (-6)), limit=10000
    ):
        """
        Compute the p-value for a chi-squared distribution using Imhof's method.

        This method computes the p-value for a given chi-squared test statistic `q`
        based on the eigenvalues `eig_vals` using Imhof's method, which is a numerical
        approach to calculate distribution functions of quadratic forms in normal variables.

        Parameters
        ----------
        q : float
            The chi-squared test statistic value.
        eig_vals : list[float]
            The eigenvalues of the matrix for which the p-value is being computed.
        epsabs : float, optional
            Absolute error tolerance for the numerical integration (default is 1e-6).
        epsrel : float, optional
            Relative error tolerance for the numerical integration (default is 1e-6).
        limit : int, optional
            The maximum number of function evaluations allowed during numerical integration (default is 10000).

        Returns
        -------
        float
            The computed p-value for the chi-squared distribution.
        """
        # matching data types to contstructor definition in C++ file
        h: np.ndarray = np.repeat(1.0, len(eig_vals))
        h: list[float] = h.tolist()
        delta: np.ndarray = np.repeat(0.0, len(eig_vals))
        delta: list[float] = delta.tolist()
        eig_vals: list[float] = eig_vals.tolist()
        q: float = float(q)
        lambda_length: int = len(eig_vals)

        # Compute the p-value for the chi-squared distribution
        p_val = imhoff.probQsupx(
            q, eig_vals, lambda_length, h, delta, epsabs, epsrel, limit
        )
        return p_val

    def spaco_test(self, x: np.ndarray) -> float:
        """
        Compute the spatially variable test statistic for any given input vector x.

        Returns
        -------
        sigma: np.ndarray
            The eigenvalues of the transformed matrix.
        """

        # Scaling input vector (should this just be centered and not scaled? )
        gene: np.ndarray = (x - x.mean()) / x.std()
        assert np.isclose(gene.mean(), 0, atol=1e-8) and np.isclose(gene.std(), 1, atol=1e-8), "Gene is not centered and scaled"

        # Compute the eigenvalues of the transformed matrix and the graph Laplacian (L)
        sorted_sigma_eigh = self.sigma_eigh[::-1]
        # printing all the variables berfore the test statistic
        
        # Normalize the scaled data
        gene = gene / np.repeat(np.sqrt(gene.T @ self.graphLaplacian @ gene), len(gene))
        print(f'sigma: {self.sigma.shape}\ngene: {gene.shape}')
        # Compute the test statistic
        test_statistic: float = float(gene.T @ self.sigma @ gene)
        # print(f'test statistic: {test_statistic}\n\n\n sorted_sigma_eigenvals: {sorted_sigma_eigh[:nSpacs]}')
        # pval test statistic
        pVal: float = self.__psum_chisq(
            q=test_statistic, eig_vals=sorted_sigma_eigh[:self.Vk.shape[1]]
        )
        print(f'pval: {pVal}\ntest statistic: {test_statistic}\nlambda cut: {self.lambda_cut}')
        return pVal, test_statistic


if __name__ == "__main__":
    None
