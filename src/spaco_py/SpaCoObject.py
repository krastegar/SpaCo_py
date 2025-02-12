import numpy as np
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler

class SPACO:
    def __init__(self, c=0.95, lambda_cut=None):
        self.c = c
        self.lambda_cut = lambda_cut
        self.Wr = None
        self.Dr = None
        self.Vk = None

    def preprocess(self, X):
        if X.shape[0] < X.shape[1]:  # Ensure X is spots × features
            X = X.T
        X = X[:, np.std(X, axis=0) > 0]  # Remove constant features
        scaler = StandardScaler()
        return scaler.fit_transform(X)

    def pca_whitening(self, X):
        XT_X = X.T @ X
        eigvals, eigvecs = eigh(XT_X)
        idx = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
        
        total_variance = np.sum(eigvals)
        cumulative_variance = np.cumsum(eigvals) / total_variance
        r = np.searchsorted(cumulative_variance, self.c) + 1
        
        self.Wr = eigvecs[:, :r]
        self.Dr = np.diag(eigvals[:r])
        return X @ self.Wr @ np.linalg.inv(np.sqrt(self.Dr))

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
