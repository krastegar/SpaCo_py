import unittest
import numpy as np
from spaco_py.SpaCoObject import SPACO

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

        Returns: -> np.ndarray
            synthetic data matrices.
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
        self.assertTrue(
            self.spaco.A.shape[0] == self.spaco.A.shape[1],
            msg="A is not a square matrix",
        )

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

        # checking to see if the output is not empty
        self.assertTrue(whitened_matrix.size > 0, msg="Output matrix is empty")

        # checking to see if there are no NaN values in the output
        self.assertFalse(
            np.any(np.isnan(whitened_matrix)), msg="NaN values present in the output"
        )

        # checking to see if the covariance matrix of the whitened matrix is close to the identity matrix
        np.testing.assert_allclose(
            whitened_matrix @ whitened_matrix.T / whitened_matrix.shape[0],
            np.eye(whitened_matrix.shape[0]),
            atol=1e-2,
        )

    def test_spectral_filtering(self):
        # checking to see if the function just works without any errors
        filter_results = ()
        filter_results = self.spaco._SPACO__spectral_filtering()

        # checking to see if all objects are returned as expected
        if len(filter_results) != 4:
            self.fail(
                "not all components of the spectral filtering function were returned"
            )

        # checking to see if all objects are numpy arrays
        _ = [
            self.assertTrue(
                isinstance(obj, np.ndarray), msg=f"object {obj} is not a numpy array"
            )
            for obj in filter_results
        ]

        # checking individual elements of the tuple
        sampled_sorted_eigvecs, sampled_sorted_eigvals, whitened_data, L = (
            filter_results
        )

        # Checking dimensions and making sure the k_cut function works
        self.assertNotEqual(
            sampled_sorted_eigvecs.shape[1],
            whitened_data.shape[1],
            msg="Number of features should not be the same after filtering",
        )

        # checking to see if eigenvalues are less than two and not negative???????
        self.assertTrue(
            np.all(sampled_sorted_eigvals <= 2) and np.all(sampled_sorted_eigvals >= 0),
            msg="Eigenvalues are not less than two and not negative",
        )

        # checking to see if the lambda cut filtering worked
        self.assertTrue(
            np.all(sampled_sorted_eigvals >= self.spaco.lambda_cut),
            msg="Eigenvalues are not greater than lambda cut",
        )

    def test_spaco_projection(self):
        # checking to see if the function just works without any errors
        Pspac, Vk, L = self.spaco.spaco_projection()

        # Checking to see if there are any NaN values in the output
        self.assertFalse(
            np.any(np.isnan(Pspac)), msg="NaN values present in the output"
        )
        self.assertFalse(np.any(np.isnan(Vk)), msg="NaN values present in the output")
        self.assertFalse(np.any(np.isnan(L)), msg="NaN values present in the output")

    def test_orthogonalize(self):
        # Debugging the orthogonalization function
        # checking to see if the function just works without any errors
        orthogonalized_matrix = self.spaco._SPACO__orthogonalize(
            self.spaco.SF, self.spaco.A, self.spaco.SF.shape[1]
        )

        # Checking to see if the matrix is actually orthogonal
        np.testing.assert_allclose(
            orthogonalized_matrix @ orthogonalized_matrix.T,
            np.eye(orthogonalized_matrix.shape[0]),
            atol=1e-2,
        )

    def test_spaco_test(self):
        # checking to see if the function works and produces something meaningful
        Pspac, Vk, L = self.spaco.spaco_projection()

        spatial_variable_feature = Pspac[:, 1]

        # the test fails due to the orthogonalization function (something about illegal square root)
        test_statistic, p_value = self.spaco.spaco_test(spatial_variable_feature)

        for test_stat, p_val in zip(test_statistic, p_value):
            print(f"Test statistic: {test_stat}, p-value: {p_val}")

    def tearDown(self):
        del self.spaco
        del self.sample_features
        del self.neighbormatrix


if __name__ == "__main__":
    unittest.main()
