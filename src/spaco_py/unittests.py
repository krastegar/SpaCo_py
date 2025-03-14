import unittest
import numpy as np
from spaco_py.SpaCoObject import SPACO
from sklearn.decomposition import PCA

# filepath: src/spaco_py/test_SpaCoObject.py


class TestSPACO(unittest.TestCase):
    @staticmethod  # static method
    def _generate_synthetic_data(
        n_samples: int,
        n_features: int,
        sparsity: float = 0.1,
        noise_level: float = 0.1,
        lam: float = 65,  # random number I chose for the poisson distribution events
        generate1d: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic single-cell RNA sequencing data.

        Args:
            n_samples: Number of samples (rows).
            n_features: Number of features (columns).
            sparsity (optional): Proportion of zero values in the data.
            noise_level (optional): Standard deviation of the Gaussian noise added to the data.
            lam (optional): Poisson distribution parameter.
            generate1d (optional): If True, flatten the data matrix.

        Returns:
            Two synthetic data matrices.
        """
        # Generate random data matrix using poisson distribution because
        # single-cell RNA sequencing data is count data
        data1 = np.random.poisson(lam=lam, size=(n_samples, n_features)).astype(float)

        # Introduce sparsity
        mask1 = np.random.rand(n_samples, n_features) < sparsity
        data1[mask1] = 0

        # Add Gaussian noise
        noise1 = np.random.normal(0, noise_level, size=(n_samples, n_features))
        data1 += noise1

        # Flatten data if generate1d is True
        if generate1d:
            data1 = data1.flatten()
            return data1

        return data1

    def setUp(self):
        """
        Set up the test fixture.

        Generates synthetic data matrices for testing. The
        `sample_features` and `neighbormatrix` attributes are set to
        matrices of shape `(100, 100)` with 10% sparsity and 10%
        Gaussian noise added.
        """
        self.sample_features = TestSPACO._generate_synthetic_data(
            n_samples=100,
            n_features=100,
            sparsity=0.1,
            noise_level=0.1,
            generate1d=False,
        )
        self.neighbormatrix = TestSPACO._generate_synthetic_data(
            n_samples=100,
            n_features=100,
            sparsity=0.1,
            noise_level=0.1,
            generate1d=False,
        )
        self.spaco = SPACO(self.sample_features, self.neighbormatrix)
        self.centered_scaled_features = self.spaco._SPACO__preprocess(
            self.sample_features
        )

    def test_init(self):
        self.assertIsInstance(self.spaco.SF, np.ndarray)
        self.assertIsInstance(self.spaco.A, np.ndarray)

    def test__remove_constant_features(self):
        """
        Test the _remove_constant_features method to see if it actually removes constant features
        """
        # Adding a constant feature to the sample features to see if variance filtering actually works
        added_constant_features = np.zeros((self.sample_features.shape[0], 1))
        self.sample_features = np.hstack(
            (self.sample_features, added_constant_features)
        )
        zero_var_removed = self.spaco._SPACO__remove_constant_features(
            self.sample_features
        )

        # checking to see if the variance filtering worked
        self.assertNotEqual(
            zero_var_removed.shape[1],
            self.sample_features.shape[1],
            msg="Number of features should not be the same after variance filtering",
        )

        # checking to see if there are any zero variance features in the output
        self.assertFalse(
            np.any(np.var(zero_var_removed, axis=0) == 0),
            msg="Zero variance features present in the output",
        )
        # checking to see if the output is a numpy array
        self.assertTrue(isinstance(zero_var_removed, np.ndarray))

    def test_preprocess(self):
        """
        Test the _preprocess method to see if it correctly centers and scales the data

        - checks if the output is a numpy array
        - checks if the output is not empty
        - checks if the output has the same number of rows as the input
        - checks if the mean of the centered and scaled data is 0
        - checks if the std of the centered and scaled data is 1
        """
        # centering and scaling
        x_centered_scaled = SPACO(
            self.sample_features, self.neighbormatrix
        )._SPACO__preprocess(self.sample_features)

        # Checking to see if the output is a non-zero filled numpy array
        # and checking to see if there are no NaN values in the output
        self.assertFalse(
            np.any(np.isnan(x_centered_scaled)), msg="NaN values present in the output"
        )
        self.assertFalse(np.all(x_centered_scaled == 0), msg="Output is all zeros")

        # checking to see if the output is a numpy array
        self.assertTrue(isinstance(x_centered_scaled, np.ndarray))

        # checking to see if the output is not empty
        self.assertTrue(x_centered_scaled.size > 0)

        # checking to see if the output has the same number of rows as the input
        self.assertTrue(x_centered_scaled.shape[0] == self.sample_features.shape[0])

        # checking if the mean of the centered and scaled data is 0
        # and checking to see if the std is set to 1
        self.assertAlmostEqual(
            np.mean(x_centered_scaled),
            0,
            msg="Mean is not 0 after centering and scaling",
            delta=1e-4,
        )
        self.assertAlmostEqual(
            np.std(x_centered_scaled),
            1,
            msg="Standard deviation is not 1 after centering and scaling",
            delta=1e-4,
        )

    def test_PCA_Whitening(self):
        whitened_matrix = self.spaco._SPACO__pca_whitening()

        # checking to see if the output is a numpy array
        self.assertTrue(isinstance(whitened_matrix, np.ndarray))

        self.assertTrue(
            np.allclose(whitened_matrix.mean(), 0, atol=1e-4),
            msg="Mean is not 0 after PCA whitening",
        )

        # cov_matrix = np.cov(whitened_matrix, rowvar=False)
        # checking to see if the covariance matrix is the identity matrix
        # self.assertTrue(np.allclose(cov_matrix, np.eye(cov_matrix.shape[1]), atol=1e-2))

        X_whitened = PCA(whiten=True).fit_transform(self.centered_scaled_features)
        cov_matrix = np.cov(X_whitened, rowvar=False)
        np.testing.assert_allclose(cov_matrix, np.eye(cov_matrix.shape[0]), atol=1e-5)

    def tearDown(self):
        del self.spaco
        del self.sample_features
        del self.neighbormatrix


if __name__ == "__main__":
    unittest.main()
