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
# 8. `fit`: Calls the `spectral_filtering` method to fit the model to the data.
# 
# 9. `transform`: Calls the `spaco_projection` method to transform the data using 
#    the fitted model.
#
# ============================================================================
# Date: 20/02/2025
# Author: Kiarash Rastegar
# ============================================================================
import numpy as np
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler

class SPACO:
    def __init__(self, sample_features, neighbormatrix,c=0.95, lambda_cut=None, percentile=95, resample_iterations=100):
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
        resample_iterations : int, optional
            The number of iterations for resampling in the eigenvalue threshold calculation. Default is 100.

        Returns
        -------
        None
        """
        self.percentile = percentile
        self.resample_iterations = resample_iterations
        self.lambda_cut = lambda_cut
        self.SF = sample_features
        self.A = neighbormatrix
        self.c = c
        
    def preprocess(self, sample_features: np.ndarray) -> np.ndarray:
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
        # Ensure X is spots × features
        if sample_features.shape[0] < sample_features.shape[1]:
            # If the number of spots is less than the number of features, transpose the array
            sample_features = sample_features.T
        
        # Remove constant features 
        sample_features = sample_features[:, np.var(sample_features, axis=0) > 0]
        # The np.var function calculates the variance of each feature,
        # and the > 0 operation filters out the features with zero variance
        
        # Scale the features using StandardScaler
        scaler = StandardScaler()
        return scaler.fit_transform(sample_features)

    def pca_whitening(self) -> np.ndarray:
        """
        Perform PCA whitening on the sample features array.

        The method first computes the covariance matrix of the sample features array.
        Then it computes the eigenvalues and eigenvectors of the covariance matrix,
        and sorts them in descending order of the eigenvalues.
        The method then selects the top r eigenvectors, where r is the number of
        eigenvectors that explain at least c percent of the variance of the data.
        The method then computes the whitened data by multiplying the sample
        features array with the selected eigenvectors and dividing by the square
        root of the eigenvalues.

        The whitening is done by computing the following:
        X_whitened = X @ Wr @ inv(sqrt(Dr))

        where Wr is an orthonormal matrix composed of the eigenvectors of the covariance matrix,
        Dr is a diagonal matrix of the eigenvalues of the covariance matrix,
        and X is the sample features array.

        Parameters
        ----------
        None

        Returns
        -------
        X_whitened : numpy array
            Whitened data matrix
        """
        # Compute the covariance matrix of the sample features array
        covariance_matrix: np.ndarray = self.SF.T @ self.SF

        # Compute the eigenvalues and eigenvectors of the covariance matrix,
        # and sort them in descending order of the eigenvalues
        eigvals: np.ndarray, eigvecs: np.ndarray = eigh(covariance_matrix)
        idx: np.ndarray = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
        
        # Select the top r eigenvectors, where r is the number of
        # eigenvectors that explain at least c percent of the variance of the data
        total_variance: float = np.sum(eigvals)
        cumulative_variance: np.ndarray = np.cumsum(eigvals) / total_variance 
        r: int = np.searchsorted(cumulative_variance, self.c) + 1
        
        # Compute the whitening matrix
        self.Wr: np.ndarray = eigvecs[:, :r] # orthonormal matrix composed of the eigenvectors of the covariance matrix
        self.Dr: np.ndarray = np.diag(eigvals[:r])
        
        # Compute the whitened data
        return self.SF @ self.Wr @ np.linalg.inv(np.sqrt(self.Dr))

    def _resample_lambda_cut(self):
        
        ## get the function that captures the largest eigenvalue (not even eigenvectors are needed)
        ## use standard L (do not permute Neighbor matrix)
        ## need to permute data (Y)

        # completely garbage (one day i learn how to code this properly)
        # resampling
        eigenvalues_list = []
        n = self.A.shape[0]
        for _ in range(self.resample_iterations):
            A_resampled = np.random.permutation(self.A)  # make sure that this is row  (spots) permutation
            L_resampled = (1 / n) * np.eye(n) + (1 / np.abs(A_resampled).sum()) * A_resampled
            M_resampled = self.Y.T @ L_resampled @ self.Y
            eigenvalues, _ = eigh(M_resampled)
            eigenvalues_list.append(eigenvalues.max())  # Store largest eigenvalue
        
        # finds the eigenvalue that corresponds to the specified percentile within the given list of eigenvalues
        return np.percentile(eigenvalues_list, self.percentile)

    def spectral_filtering(self):
        # whiten the data 
        whitened_data = self.pca_whitening()
    
        # calculating number of samples
        n = self.A.shape[0]

        # calculating lambda cut
        lambda_cut = self._resample_lambda_cut() 
        
        # calculating graph laplacian matrix
        L = (1/n) * np.eye(n) + (1/np.abs(self.A).sum()) * self.A
        
        # constructing M matrix
        M = whitened_data.T @ L @ whitened_data
        
        # Eigen decomposition of M and sorting of eigenvalues and eigenvectors
        eigvals, eigvecs = eigh(M)
        idx = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
        
        # k is the index of the last eigen value that is greater than or equal to  lambda_cut
        k = np.searchsorted(eigvals, lambda_cut) + 1 
        sampled_sorted_eigvecs = eigvecs[:, :k]
        sampled_sorted_eigvals = eigvals[:k]
        return (sampled_sorted_eigvecs, sampled_sorted_eigvals, whitened_data, L)

    def spaco_projection(self):
        # getting the sorted and reduced most relevant eigenvalues and eigen vectors from spectral filtering
        sampled_sorted_eigvecs, sampled_sorted_eigvals, whitened_data, L = self.spectral_filtering()


        # Generating orthonormal matrix in SPACO space
        U = sampled_sorted_eigvecs / np.sqrt(sampled_sorted_eigvals)
        Vk = whitened_data @ U

        return Vk @ Vk.T @ L @ self.SF

    def spaco_test(self, x, L):
        sigma = np.linalg.eigvalsh(self.Vk.T @ L @ L @ self.Vk)
        test_statistic = np.linalg.norm(self.Vk.T @ L @ x)**2
        return test_statistic, sigma

    def fit(self):
        self.spectral_filtering()
        return self

    def transform(self):
        return self.spaco_projection()
