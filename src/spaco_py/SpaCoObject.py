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
import spaco_py.imhoff as imhoff
from scipy.linalg import eigh
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class SPACO:
    def __init__(
        self, sample_features, neighbormatrix, c=0.95, lambda_cut=None, percentile=95
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
        self.SF: np.ndarray = self.__preprocess(sample_features)
        self.A: np.ndarray = self.__check_if_square(neighbormatrix)
        self.c: float = c

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
        # Initialize StandardScaler
        scaler = StandardScaler()

        # checking to see if the input is a vector
        X = self.__remove_constant_features(X)

        # returning scaled and centered features (using z-scaling)
        return scaler.fit_transform(X)

    def __check_if_square(self, X: np.ndarray) -> bool:
        """
        Check if the input data is a square matrix.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        """
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
        of the columns of the projection matrix X and returns a Q.

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

    def __pca_whitening(self) -> np.ndarray:
        """
        Applies PCA (Prinzation that Achim, David and Niklaus came up with
        for the SPACO acipal Component Analysis) whitening to the stored feature matrix.
        PCA whitening decorrelates the features and scales them to have unit variance.
        Returns:
            np.ndarray: The transformed feature matrix after PCA whitening.
        """
        # Declaring variables
        x: np.ndarray = self.SF

        # checking to see if the input data is actually centered and scaled
        if round(x.mean(), 2) != 0 and round(x.std(), 2) != 1:
            raise ValueError("The input data is not centered.")

        # np.allclose(x.std(axis=1), np.ones(len(unique_sums)), atol=1e-4)
        # PCA whitening sklearn
        pca = PCA(whiten=True)

        return pca.fit_transform(x)

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
        whitened_data : numpy.ndarray
            The whitened data.
        L : numpy.ndarray
            The computed graph Laplacian.
        """
        # Declaring variables
        whitened_data: np.ndarray = self.__pca_whitening()
        neighbor_matrix: np.ndarray = self.A
        eigvals: np.ndarray
        eigvecs: np.ndarray

        # Calculating Graph Laplacian and M matrix
        n: int = neighbor_matrix.shape[0]
        L: np.ndarray = (1 / n) * np.eye(n) + (
            1 / np.abs(neighbor_matrix).sum()
        ) * neighbor_matrix
        M: np.ndarray = whitened_data.T @ L @ whitened_data
        if M.shape[0] != M.shape[1]:
            raise ValueError("M matrix is not square issue with matrix multiplication")
        # Eigenvalue decomposition
        eigvals, eigvecs = eigh(M)
        idx: np.ndarray = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

        # Simple heuristic method for lambda cut (for now just median...later resampling)
        if self.lambda_cut is None:
            self.lambda_cut: float = np.median(eigvals)

        k: int = len(eigvals[eigvals >= self.lambda_cut])  # index where lambda cut is
        sampled_sorted_eigvecs: np.ndarray = eigvecs[:, :k]
        sampled_sorted_eigvals: np.ndarray = eigvals[:k]

        return (sampled_sorted_eigvecs, sampled_sorted_eigvals, whitened_data, L)

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
        U: np.ndarray
        Vk: np.ndarray
        Pspac: np.ndarray
        sample_feature_matrix: np.ndarray

        sample_feature_matrix = self.SF

        # getting the sorted and reduced most relevant eigenvalues and eigen vectors from spectral filtering
        sampled_sorted_eigvecs, sampled_sorted_eigvals, whitened_data, L = (
            self.__spectral_filtering()
        )

        # Generating orthonormal matrix in SPACO space
        U = sampled_sorted_eigvecs / np.sqrt(sampled_sorted_eigvals)
        # print(f'First few vecs of view: {U[:, :5]}')
        Vk = whitened_data @ U
        Pspac = Vk @ Vk.T @ L @ sample_feature_matrix
        return Pspac, Vk, L

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
        # Declaring variables
        Vk: np.ndarray
        L: np.ndarray
        _, Vk, L = self.spaco_projection()

        # Compute the number of SPACO components
        nSpacs: int = Vk.shape[1]

        # Compute the matrix Sk by taking the first nSpacs - 1 columns of the projection matrix Pspac
        projection: np.ndarray = self.__orthogonalize(Vk, self.A, nSpacs)
        Sk: np.ndarray = projection[
            :, :nSpacs
        ]  # in the R code, it is S = projection[, 1:nSpacs]

        # Compute the transformed matrix L @ Sk @ Sk.T @ L
        sigma: np.ndarray = Sk.T @ L @ L @ Sk  # --L @ Sk @ Sk.T @ L

        # Compute the eigenvalues of the sigma matrix
        sigma_eigh: np.ndarray = np.linalg.eigvalsh(sigma)

        return sigma_eigh, L, sigma, nSpacs

    def __psum_chisq(
        self, q, eig_vals, epsabs=10 ^ (-6), epsrel=10 ^ (-6), limit=10000
    ):
        h = (np.repeat(1, len(eig_vals)),)
        delta = (np.repeat(0, len(eig_vals)),)

        # Compute the p-value for the chi-squared distribution
        print(
            "Computing p-value for the weighted sum of chi-squared distribution, using imhoff algorithm...."
        )
        p_val = imhoff.probQsupx(q, eig_vals, h, delta, epsabs, epsrel, limit)
        return p_val

    def spaco_test(self, x: np.ndarray) -> float:
        """
        Compute the spatially variable test statistic for any given input vector x.

        Returns
        -------
        sigma: np.ndarray
            The eigenvalues of the transformed matrix.
        """
        # Declaring variables
        L: np.ndarray
        sigma_eigh: np.ndarray
        sigma: np.ndarray
        nSpacs: int

        # Scaling input vector (should this just be centered and not scaled? )
        gene: np.ndarray = (x - x.mean()) / x.std()

        # Compute the eigenvalues of the transformed matrix and the graph Laplacian (L)
        sigma_eigh, L, sigma, nSpacs = self.__sigma_eigenvalues()

        # Normalize the scaled data
        gene = gene / np.repeat(np.sqrt(gene.T @ L @ gene), len(gene))

        # Compute the test statistic
        test_statistic: float = gene.T @ sigma @ gene

        # pval test statistic
        pVal: float = self.__psum_chisq(
            q=test_statistic, eig_vals=sigma_eigh[:nSpacs], df=np.repeat(1, nSpacs)
        )

        return pVal, test_statistic


if __name__ == "__main__":
    x = np.linspace(0, 100, 10)
    None
