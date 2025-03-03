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
from scipy.linalg import eigh
from typing import Tuple
from sklearn.preprocessing import StandardScaler


class SPACO:
    def __init__(
        self, sample_features, neighbormatrix, c=0.95, lambda_cut=90, percentile=95
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
        self.percentile: int = percentile
        self.lambda_cut: int = lambda_cut
        self.SF: np.ndarray = self._preprocess(sample_features)
        self.A: np.ndarray = neighbormatrix
        self.c: float = c

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        """
        Preprocess the sample features array.

        The preprocessing steps are:
        1. Ensure X is spots × features
        2. Remove constant features
        3. Scale the features using StandardScaler

        Args:
            sample_features: Spots × Features array to be preprocessed

        Returns:
            Preprocessed Spots × Features array
        """
        # Ensure X is spots × features by ensuring dimensions would match theory
        # -------THIS ACTUALLY MIGHT BE UNNECESSARY and COULD CAUSE A bug------- #
        if X.shape[0] < X.shape[1]:
            X = X.T
        # -------THIS ACTUALLY MIGHT BE UNNECESSARY and COULD CAUSE A bug------- #

        # Remove constant features
        X = self.SF[:, np.var(X, axis=0) > 0]

        # returning scaled and centered features (using z-scaling)
        scaler = StandardScaler()
        return scaler.fit_transform(X)

    def _orthogonalize(self, X, A, nSpacs):
        """
        Version of QR factorization that Achim, David and Niklaus came up with
        for the SPACO algorithm. Want to talk to achim about replacing this whole thing
        just with np.linalg.qr() function. This implents a gram-schmidt orthogonalization
        of the columns of the projection matrix X and returns a Q.
        """

        # preFactor = 1 # not sure what this is for

        # getting number of rows and columns of the projection matrix X
        m = X.shape[0]
        n = X.shape[1]
        if m < n:
            raise ValueError(
                "The number of rows of the projection matrix must be greater than or equal to the number of columns."
            )
        Q: np.ndarray = np.zeros((m, n))  # initializing the orthogonalized matrix
        norms: np.ndarray = np.zeros(
            n
        )  # initializing the norms of the columns of the projection matrix (this is a vector)
        for k in range(1, nSpacs):
            Q[:, k] = X[
                :, k
            ]  # setting the k-th column of the orthogonalized matrix to the k-th column of the projection matrix
            if k > 1:
                for i in range(1, k - 1):
                    repeated_value: np.ndarray = np.repeat(
                        (Q[:, k]) @ A @ Q[:, i] / (Q[:, i] @ A @ Q[:, i]), m
                    )
                    Q[:, k] = Q[:, k] - repeated_value * Q[:, i]
            norms[k] = np.sqrt(Q[:, k] @ A @ Q[:, k])
            Q[:, k] = Q[:, k] / norms[k]  # not sure why in R version its c(Norms[k])
        return Q

    def _pca_whitening(self):
        # This function is most likely going to be replaced by sklearn pca function, with whitening set to true
        # center data and scale data
        centered_scaled = self.SF

        # Generating Covariance Matrix and Eigenvalue decomposition
        covariance_matrix = centered_scaled.T @ centered_scaled
        eigvals, eigvecs = eigh(covariance_matrix)
        idx = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

        # Select the top r eigenvectors, where r is the number of
        # eigenvectors that explain at least c percent of the variance of the data
        total_variance: float = np.sum(eigvals)
        cumulative_variance: np.ndarray = np.cumsum(eigvals) / total_variance
        r: int = np.searchsorted(cumulative_variance, self.c) + 1

        # Compute the whitening matrix
        self.Wr: np.ndarray = eigvecs[
            :, :r
        ]  # orthonormal matrix composed of the eigenvectors of the covariance matrix
        self.Dr: np.ndarray = np.diag(eigvals[:r])

        # Compute the whitened data
        total_variance = np.sum(eigvals)
        cumulative_variance = np.cumsum(eigvals) / total_variance
        r = np.searchsorted(cumulative_variance, self.c) + 1

        self.Wr = eigvecs[:, :r]
        self.Dr = np.diag(eigvals[:r])
        return self.SF @ self.Wr @ np.linalg.inv(np.sqrt(self.Dr))

    def _spectral_filtering(
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
        Y : numpy.ndarray
            The whitened data.
        L : numpy.ndarray
            The computed graph Laplacian.
        """
        # Declaring variables
        Y: np.ndarray = self._pca_whitening()
        A: np.ndarray = self.A
        eigvals: np.ndarray
        eigvecs: np.ndarray

        # Calculating Graph Laplacian and M matrix
        n: int = A.shape[0]
        L: np.ndarray = (1 / n) * np.eye(n) + (
            1 / np.abs(A).sum(axis=0)
        ) * A  # axis=0 -> rows
        M: np.ndarray = Y.T @ L @ Y

        # Eigenvalue decomposition
        eigvals, eigvecs = eigh(M)
        idx: np.ndarray = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

        # Simple heuristic method for lambda cut (for now just median...later resampling)
        if self.lambda_cut is None:
            self.lambda_cut: float = np.median(eigvals)

        # k is the index of the last eigen value that is greater than or equal to  lambda_cut
        k: int = np.searchsorted(eigvals, self.lambda_cut) + 1
        sampled_sorted_eigvecs: np.ndarray = eigvecs[:, :k]
        sampled_sorted_eigvals: np.ndarray = eigvals[:k]
        return (sampled_sorted_eigvecs, sampled_sorted_eigvals, Y, L)

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
        sample_feature_matrix: np.ndarray = self.SF

        # getting the sorted and reduced most relevant eigenvalues and eigen vectors from spectral filtering
        sampled_sorted_eigvecs, sampled_sorted_eigvals, whitened_data, L = (
            self._spectral_filtering()
        )

        # Generating orthonormal matrix in SPACO space
        U: np.ndarray = sampled_sorted_eigvecs / np.sqrt(sampled_sorted_eigvals)
        Vk: np.ndarray = whitened_data @ U
        Pspac: np.ndarray = Vk @ Vk.T @ L @ sample_feature_matrix
        return Pspac, Vk, L

    def spaco_test(self):
        """
        Perform a statistical test on the input data with SPACO.

        This method computes a test statistic and its associated eigenvalues
        for a given input sample and graph Laplacian matrix using the SPACO
        framework. The test statistic is based on the norm of the projection
        of the sample data onto the SPACO space, and the eigenvalues are
        derived from the double application of the graph Laplacian to the
        SPACO projection matrix.

        Returns
        -------
        test_statistic : float
            The calculated test statistic for the input sample.
        sigma : np.ndarray
            The eigenvalues of the SPACO projection matrix with respect to
            the graph Laplacian, shape (n_samples,).
        """

        Pspac, Vk, L = self.spaco_projection()

        # using np.linalg.eigvalsh() to get the eigenvalues only
        sigma: np.ndarray = np.linalg.eigvalsh(Vk.T @ L @ L @ Vk)

        # test_statistic : float = np.linalg.norm(self.Vk.T @ L @ x) ** 2

        return sigma  # for now we keep this here for the pre-commit to work
