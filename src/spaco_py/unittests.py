import unittest
import numpy as np
import pandas as pd
from spaco_py.SpaCoObject import SPACO

# filepath: src/spaco_py/test_SpaCoObject.py


class TestSPACO(unittest.TestCase):
    def setUp(self):
        """
        Using benchmark data from the SPACO paper
        """
        self.sample_features = np.load(
            "/home/krastega0/SpaCo_py/src/spaco_py/sf_mat.npy"
        )
        self.neighbormatrix = np.load("/home/krastega0/SpaCo_py/src/spaco_py/A_mat.npy")
        self.spaco = SPACO(self.sample_features, self.neighbormatrix)

        # variable is redundant, but I want to keep it for testing purposes
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

        # is Q orthogonal to the graph Laplacian?
        self.assertTrue(
            np.allclose(Q.T @ L @ Q, np.zeros((Q.shape[1], Q.shape[1])), atol=1e-2),
            msg="Q is not orthogonal to the graph Laplacian",
        )

        # checking to see if vk is positive semi-definite
        self.assertTrue(
            np.all(np.linalg.eigvalsh(Vk) >= 0),
            msg="Ortho matrix is not a positive semi-definite matrix",
        )

        # checking to see if the orthogonalization produces non-zero eigenvalues
        self.assertTrue(
            np.all(np.linalg.eigvalsh(Q) >= 0),
            msg="Ortho matrix is not a positive semi-definite matrix",
        )

    def test_sigma_eigenvalues(self):
        # Making sure that the sigma eigenvalues are computed correctly
        # And lie within the correct range
        print("Testing the __sigma_eigenvalues method")
        sigma_eigh, sigma = self.spaco._SPACO__sigma_eigenvalues()

        # sigma should be a PSD matrix
        self.assertTrue(
            np.all(np.linalg.eigvals(sigma) >= 0),
            msg="Sigma is not a positive semi-definite matrix",
        )

        # Eigenvalues should be between 0 and 1
        self.assertTrue(
            np.all(sigma_eigh >= 0) and np.all(sigma_eigh <= 1),
            msg="Eigenvalues are not between 0 and 1",
        )

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
    unittest.main(defaultTest="TestSPACO.test_orthogonalize")
    unittest.main(defaultTest="TestSPACO.test_sigma_eigenvalues")
