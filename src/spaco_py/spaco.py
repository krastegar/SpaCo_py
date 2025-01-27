import tensorflow as tf
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def pca_whitening(X, c=0.95):
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

    X_centered = (
        X.T - np.mean(X, axis=1)
    )  # column mean # only transposing for this specific data include check to see if its samples vs Features
    X_centered = X_centered.T
    # X_centered = X.T - np.mean(X.T, axis=1) # column mean
    print(f"Dimensions of centered data: {X_centered.shape}")
    column_sums = np.sum(X_centered, axis=1)

    print(f"Col sums of X_centered: {np.unique(column_sums)}")
    # Step 2: Compute the covariance matrix
    # The covariance matrix is a square, symmetric matrix where the
    # element at row i and column j is the covariance of the i-th and
    # j-th features of the dataset.
    # The covariance matrix is a measure of how much the data varies in
    # each direction. It's a measure of how spread out the data is.
    cov = np.cov(X_centered, rowvar=False)
    print(f"Dimensions of covariance matrix: {cov.shape}")

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
    X_whitened = np.dot(X_centered, W_r).dot(D_r_inv_sqrt)
    print(f"Dimensions of whitened data: {X_whitened.shape}")

    return X_whitened


if __name__ == "__main__":
    # Load the full MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Specify the desired number of samples
    num_samples = 300

    # Create StratifiedShuffleSplit object
    sss = StratifiedShuffleSplit(n_splits=1, test_size=num_samples, random_state=42)

    # Get the indices for the subset
    for train_index, subset_index in sss.split(x_train, y_train):
        x_train = x_train[subset_index]
        y_train = y_train[subset_index]

    # Reshape subset images back to original shape
    x_train = x_train.reshape(x_train.shape[0], 28, 28)

    # Now you have x_subset and y_subset with the desired number of sample
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    # Flatten the images
    x_train_flattened = x_train.reshape(x_train.shape[0], -1)

    # Data Preprocessing # -- my whitening function is extremely memory inefficient ------ #
    whitened_data = pca_whitening(x_train_flattened)
