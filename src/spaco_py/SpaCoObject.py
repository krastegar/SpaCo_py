import numpy as np
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler

class SPACO:
    def __init__(self, sample_features, neighbormatrix,c=0.95, lambda_cut=None):
        self.lambda_cut = lambda_cut
        self.SF = sample_features
        self.A = neighbormatrix
        self.c = c
        
    def preprocess(self):
        if self.SF.shape[0] < self.SF.shape[1]:  # Ensure X is spots × features
            self.SF = self.SF.T
        self.SF = self.SF[:, np.std(self.SF, axis=0) > 0]  # Remove constant features (not sure if this is correct, should we just look at variance or std?)
        scaler = StandardScaler() # class object to center and scale the data 
        return scaler.fit_transform(self.SF)

    def pca_whitening(self):
        covariance_matrix = self.SF.T @ self.SF
        eigvals, eigvecs = eigh(covariance_matrix)
        idx = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
        
        total_variance = np.sum(eigvals)
        cumulative_variance = np.cumsum(eigvals) / total_variance
        r = np.searchsorted(cumulative_variance, self.c) + 1
        
        self.Wr = eigvecs[:, :r]
        self.Dr = np.diag(eigvals[:r])
        return self.SF @ self.Wr @ np.linalg.inv(np.sqrt(self.Dr))

    def spectral_filtering(self, Y, A):
        n = A.shape[0]
        L = (1/n) * np.eye(n) + (1/np.abs(A).sum()) * A
        M = Y.T @ L @ Y
        eigvals, eigvecs = eigh(M)
        idx = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
        
        if self.lambda_cut is None:
            self.lambda_cut = np.median(eigvals)  # Simple heuristic
        k = np.searchsorted(eigvals, self.lambda_cut, side='right')
        
        Uk = eigvecs[:, :k]
        lambdak = eigvals[:k]
        self.Vk = Y @ (Uk / np.sqrt(lambdak))

    def spaco_projection(self, x):
        return self.Vk @ self.Vk.T @ x

    def spaco_test(self, x, L):
        sigma = np.linalg.eigvalsh(self.Vk.T @ L @ L @ self.Vk)
        test_statistic = np.linalg.norm(self.Vk.T @ L @ x)**2
        return test_statistic, sigma

    def fit(self, X, A):
        X = self.preprocess(X)
        Y = self.pca_whitening(X)
        self.spectral_filtering(Y, A)
        return self

    def transform(self, X):
        X = self.preprocess(X)
        Y = X @ self.Wr @ np.linalg.inv(np.sqrt(self.Dr))
        return self.spaco_projection(Y)
