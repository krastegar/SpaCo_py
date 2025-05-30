import unittest
import numpy as np
import pandas as pd
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
            n_samples=80,
            n_features=100,
            sparsity=0.1,
            noise_level=0.1,
            generate1d=False,
        )
        self.neighbormatrix = TestSPACO._generate_synthetic_data(
            n_samples=80,
            n_features=80,
            sparsity=0.1,
            noise_level=0.1,
            generate1d=False,
        )
        self.spaco = SPACO(self.sample_features, self.neighbormatrix)
        self.centered_scaled_features = self.spaco._SPACO__preprocess(
            self.sample_features
        )

    def test_init(self):
        print("Testing the __init__ method")
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
        print("Testing the __remove_constant_features method")
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
        print("Testing the __preprocess method")
        # centering and scaling
        x_centered_scaled = SPACO(
            self.sample_features, self.neighbormatrix
        )._SPACO__preprocess(self.sample_features)

        # checking to see if the condition I put to check if the data is a numpy array
        # is actually working
        sf_df = pd.DataFrame(self.sample_features)

        # checking to see if the change in data type from numpy array to pandas dataframe
        # does not cause an error
        SPACO(sf_df, self.neighbormatrix)._SPACO__preprocess(sf_df)

        # checking to see if the output is a numpy array
        self.assertTrue(isinstance(x_centered_scaled, np.ndarray))

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
        print("Testing the __pca_whitening method")
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
            whitened_matrix.T @ whitened_matrix / whitened_matrix.shape[0],
            np.eye(whitened_matrix.shape[1]),
            atol=0.1,
        )

    def test_shuffle_decomp(self):
        # creating a new object with mouse brain data and maybe it will work

        # plan of attack:
        print("\nTesting the __shuffle_decomp method\n")
        # running it once by itself to see if it works
        print("Standalone attempt at running shuffle_decomp")
        self.spaco._SPACO__shuffle_decomp()

        # running it in a for loop to see if its just the shuffle portion that is breaking it
        for i in range(1, 1000):
            print(f"Attempting to run shuffle_decomp {i} times")
            try:
                largest_eigval = self.spaco._SPACO__shuffle_decomp()
                print(f"largest_eigval: {largest_eigval}")
            except Exception as e:
                print(
                    "error: most likely a non-convergence error, not numerically stable\n\n"
                )
                print(f"error message: {e}")
                self.fail(f"shuffle_decomp failed after {i} iterations")
        else:
            print("Shuffle decomposition completed successfully.")
        return

    def test__resample_lambda_cut(self):
        print("Testing the __resample_lambda_cut method")
        # same plan of attack as above
        # running it once by itself to see if it works
        self.spaco._SPACO__resample_lambda_cut()
        return

    def test_spectral_filtering(self):
        print("Testing the __spectral_filtering method")
        # checking to see if the function just works without any errors
        filter_results = self.spaco._SPACO__spectral_filtering()

        # checking to see if all objects are returned as expected
        if len(filter_results) != 3:
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
        sampled_sorted_eigvecs, sampled_sorted_eigvals, _ = filter_results

        # Checking dimensions and making sure the k_cut function works
        self.assertNotEqual(
            sampled_sorted_eigvecs.shape[1],
            self.spaco.whitened_data.shape[1],
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
        print("Testing the spaco_projection method")
        # checking to see if the function just works without any errors
        _, Vk = self.spaco.spaco_projection()

        k: int = len(
            self.spaco.sampled_sorted_eigvals
        )  # the dimemsionally reduced data

        # checking to see if the whitened data contains negative entries
        # the both eigvecs and whitened data are allowed to have negative entries
        n: int = self.spaco.SF.shape[0]  # number of samples

        # matrix for spaco embeddings is the correct shape
        self.assertEqual(Vk.shape, (n, k))

        # Make sure that L matrix is the correct shape
        self.assertEqual(
            self.spaco.graphLaplacian.shape,
            (self.spaco.whitened_data.shape[0], self.spaco.whitened_data.shape[0]),
        )

        # make sure that the U matrix that makes up Vk is orthonormal
        U = self.spaco.sampled_sorted_eigvecs / np.sqrt(
            self.spaco.sampled_sorted_eigvals
        )

        # U is orthonormal
        np.allclose(np.eye(U.shape[0]), U @ U.T, atol=1e-2)

        # Vk should still have the whitening properties
        cov_vk = np.cov(Vk)

        # checking to see if the covariance matrix of vk is the identity and has mean 0
        np.allclose(cov_vk / Vk.shape[0], np.eye(Vk.shape[0]), atol=1e-2)
        self.assertAlmostEqual(
            np.mean(Vk),
            0,
            places=1,
            msg="the mean of the projections in spaco space is not 0",
        )
        # checking the symmetry of graph laplacian
        if np.allclose(self.neighbormatrix, self.neighbormatrix.T, atol=1e-2):
            np.allclose(
                self.spaco.graphLaplacian, self.spaco.graphLaplacian.T, atol=1e-2
            )

    def test_orthogonalize(self):
        print("Testing the __orthogonalize method")
        Vk, L = self.spaco.Vk, self.spaco.graphLaplacian
        # print(f'shape of Neighborhood: {self.neighbormatrix.shape}\n\n shape of Vk: {Vk.shape}\n\n shape of L: {L.shape}')

        # takes Vk as the inner product with A
        Q = self.spaco._SPACO__orthogonalize(X=Vk, A=L, nSpacs=Vk.shape[1])

        # print(f'shape of Q: {Q.shape}')
        # check to see that Q is orthogonal
        np.allclose(Q.T @ L @ Q, np.eye(Q.shape[1]), atol=1e-5)

    def test_spaco_test(self):
        print("Testing the spaco_test method")
        for i in range(self.spaco.Pspac.shape[1]):
            pval, t = self.spaco.spaco_test(self.spaco.Pspac[:, i])

            # want to make sure that pval is between 0 and 1
            self.assertTrue(pval >= 0 and pval <= 1, msg="pval is not between 0 and 1")

            # want to make sure that t is between 0 and 2
            self.assertTrue(t >= 0 and t <= 2, msg="t is not between 0 and 2")

            # print(f"pval: {pval}, spatial relevance score: {t}")

    def tearDown(self):
        del self.spaco
        del self.sample_features
        del self.neighbormatrix


if __name__ == "__main__":
    unittest.main(defaultTest="TestSPACO.test_spectral_filtering")
    # unittest.main(defaultTest="TestSPACO.test_spaco_test")
