import unittest
import numpy as np
import pandas as pd
from spaco_py.SpaCoObject import SPACO
from scipy.stats import ks_2samp
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

    def check_mark(self):
        """
        A simple method to visually show completion of the tests by printing a check mark
        """
        print("\n✅\n")

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

    def test_replicate(self):
        print("testing the __replicate method")

        # going to test 3 different replicate numbers in increasing order and see if the array expands as expected
        replicate_numbers = [2, 4, 6]
        initial_size = 0

        # stack structure to keep track of the length of the array after each replicate call
        stack = [initial_size]
        for replicate_number in replicate_numbers:
            prev_len = stack[-1]
            if prev_len == 0:
                # pop stack to remove the initial size
                stack.pop()
                # first time running the replicate function
                print(f"Replicating {replicate_number} times for the first time")
                stack.append(len(self.spaco._SPACO__replicate(replicate_number)))
                continue  # go to the next iteration (skip the rest of the loop)

            # if the stack is not empty, we can check the previous length
            # checking to see if the previous length is not None
            self.assertIsNotNone(prev_len, msg="Previous length is None")
            print(f"Replicating {replicate_number} times")

            # calling the replicate function
            # checking to see if the length of the stack has increased
            self.assertGreater(
                len(self.spaco._SPACO__replicate(replicate_number)),
                prev_len,
                msg=f"Length of stack did not increase after replicating {replicate_number} times",
            )
            stack.pop()  # remove the previous length
            stack.append(len(self.spaco._SPACO__replicate(replicate_number)))

    def test__resample_lambda_cut(self):
        print("Testing the __resample_lambda_cut method")

        # running the spectral_filtering method to see if it works without any errors
        # and to extract the initial non-random eigenvalues for lambda cut calculation
        _ = self.spaco._SPACO__spectral_filtering()
        non_random_eigvals = self.spaco._cache["non_random_eigvals"]

        # initialize number of iterations
        iterations = 0

        # increase list of results by a batch size
        batch_size = 10

        # maximum number of iterations to run the loop
        n_simulations = 5

        # ----------Overall goal of the loop -----------------#
        # The goal of the loop is to iteratively reduce the confidence interval (CI) until
        # there is only one eigenvalue (lambda) left in the CI, which is the lambda_cut.
        # The loop will continue until either the CI is reduced to a single lambda or
        # the maximum number of iterations is reached.
        # ---------------------------------------------------#

        # initializing top random eigenvalues
        top_random_eigvals = np.linspace(1.3, 2, 10)

        # calculating the 95% CI and SE for the top random eigenvalues
        # should have more than one element in the list
        ci_lower_sim, ci_upper_sim = self.spaco._SPACO__CI_SE(list(top_random_eigvals))

        # calculating width of the CI
        delta_init = ci_upper_sim - ci_lower_sim

        # initializing the delta stack with the initial delta value to track the changes in delta
        # and see if the delta value decreases after each iteration
        delta_stack = [delta_init]

        # should have more than one lambda in the CI
        lambdas_inCI_sim = non_random_eigvals[
            (non_random_eigvals >= ci_lower_sim) & (non_random_eigvals <= ci_upper_sim)
        ]

        assert len(lambdas_inCI_sim) > 1, (
            "There should be more than one lambda in the CI at the start"
        )

        while len(lambdas_inCI_sim) > 1:
            delta_init = round(delta_stack[-1], 2)  # get the last delta value

            # increasing the counter
            iterations += 1

            print(f"\nIteration {iterations} with delta value: {delta_init:.3g}")

            # Adding the batch size to the number of iterations to steadily decrease the CI margins
            batch_results = self.spaco._SPACO__replicate(batch_size)

            assert len(batch_results) == batch_size, (
                "The number of results is not equal to the batch size"
            )

            # Calculate the 95% CI and SE for the new batch of results
            top_random_eigvals = np.append(top_random_eigvals, batch_results)

            # Calculate the 95% CI and SE for the new batch of results
            ci_lower, ci_upper = self.spaco._SPACO__CI_SE(list(top_random_eigvals))

            # checking to see how many lambdas are in the CI
            lambdas_inCI = non_random_eigvals[
                (non_random_eigvals >= ci_lower) & (non_random_eigvals <= ci_upper)
            ]

            # updated delta value should be smaller than the previous delta value
            delta = round(ci_upper - ci_lower, 2)
            self.assertLessEqual(
                delta, delta_init, msg="Delta value did not decrease after iteration"
            )

            # append the new delta value to the stack
            delta_stack.pop()  # remove the last delta value
            delta_stack.append(delta)

            # printing the number of iterations, the number of elements in the CI, and the size of the CI
            print(
                f"\nnumber of iterations: {iterations}.\n # of elements in CI: {len(lambdas_inCI)}"
            )
            if iterations >= n_simulations:
                # if we reach the maximum number of iterations, we break the loop
                self.check_mark()
                print(
                    f"\nReached maximum number of iterations: {n_simulations}.\n # of elements in CI: {len(lambdas_inCI)}"
                )
                break

            if len(lambdas_inCI) < 2:
                # lambdas_inCI should be a 1D array with only one element, which is the lambda of interest
                self.lambda_cut = (
                    lambdas_inCI[::-1][0] if len(lambdas_inCI) > 0 else ci_upper
                )

                # printing the lambda cut and the number of iterations
                self.check_mark()
                print(
                    f"\nPopped out all of the values from the CI. Lambda cut: {self.lambda_cut}.\n # of iterations: {iterations}.\n"
                )
                break

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
        self.assertIsNotNone(
            self.spaco.graphLaplacian,
            msg="graphLaplacian is None; it should be initialized before checking its shape",
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
        if self.spaco.graphLaplacian is not None:
            self.assertTrue(
                np.allclose(
                    self.spaco.graphLaplacian, self.spaco.graphLaplacian.T, atol=1e-2
                ),
                msg="graph Laplacian is not symmetric",
            )
        else:
            self.fail(
                "graphLaplacian is None; it should be initialized before checking"
            )

    def test_orthogonalize(self):
        print("Testing the __orthogonalize method")
        Vk, L = self.spaco.Vk, self.spaco.graphLaplacian
        # print(f'shape of Neighborhood: {self.neighbormatrix.shape}\n\n shape of Vk: {Vk.shape}\n\n shape of L: {L.shape}')

        # takes Vk as the inner product with graph Laplacian
        Q = self.spaco._SPACO__orthogonalize(V=Vk, L=L)
        identity_L = Q.T @ L @ Q

        # is Q orthogonal to the graph Laplacian?
        self.assertTrue(
            np.allclose(identity_L, np.eye(identity_L.shape[0]), atol=1e-8),
            msg="Q is not orthogonal to the graph Laplacian",
        )

    def test_sigma_eigenvalues(self):
        # Making sure that the sigma eigenvalues are computed correctly
        # And lie within the correct range
        print("Testing the __sigma_eigenvalues method")
        sigma_eigh, sigma = self.spaco._SPACO__sigma_eigenvalues()

        # need to make sure that sigma is kxk matrix
        self.assertTrue(
            sigma.shape == (self.spaco.Vk.shape[1], self.spaco.Vk.shape[1]),
            msg="Sigma is not a square matrix with k x k dimensions",
        )
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

        # generate null distribution and compare it against distribution of test statistics for random features
        # using the ks_2samp test to compare the distributions
        theoretical_null = [
            sigma_eigh.T @ np.random.chisquare(df=1, size=self.spaco.Vk.shape[1])
            for i in range(10000)
        ]

        # generate the test statistic for this random simulated pattern
        rand_test_statistic = []
        for i in range(10000):
            rand_projection = (
                np.random.normal(size=self.spaco.SF.shape[0]).T
                @ self.spaco.graphLaplacian
                @ self.spaco.Vk
            )
            len_proj = rand_projection.T @ rand_projection
            rand_test_statistic.append(len_proj)

        # compare the two distributions
        _, p_value = ks_2samp(theoretical_null, rand_test_statistic)

        # p_value should be greater than 0.05
        if p_value < 0.05:
            self.fail(
                "The null distribution is not consistent with the distribution of test statistics for random features"
            )
        else:
            self.check_mark()

    def test_spaco_test(self):
        print("Testing the spaco_test method")
        for i in range(self.spaco.Pspac.shape[1]):
            print(f"Testing spatial relevance score for feature {i}")
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
    # unittest.main(defaultTest="TestSPACO.test_spaco_test")
    unittest.main(defaultTest="TestSPACO.test_sigma_eigenvalues")
