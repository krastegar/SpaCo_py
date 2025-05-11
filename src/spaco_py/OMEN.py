from spaco_py.SpaCoObject import SPACO
import numpy as np
import time


class OMEN(SPACO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _eigen_cutoff(self):
        """
        Function to determine the cutoff for eigenvalues, for the spectral filtering
        method that is used in SPACO.
        """

        # creating a range of values to be put into replicate function from SPACO
        value_range = np.arange(10, 21, 1)

        # index for finding the middle of the range
        index = int(len(value_range) / 2)

        # call the replicate function from SPACO
        self.replicate(value_range[index])


if __name__ == "__main__":
    # Loading .npy file
    sf = np.load(
        "/home/krastegar0/SpaCo_py/src/spaco_py/sf_mat.npy", allow_pickle=False
    )
    neighbor = np.load(
        "/home/krastegar0/SpaCo_py/src/spaco_py/A_mat.npy", allow_pickle=False
    )

    # calling SPACO class
    omen = OMEN(sample_features=sf, neighbormatrix=neighbor)
    # checking the keys of the loaded files
    # print(type(sf), type(neighbor))

    # value range for the replicate function
    value_range = np.arange(100, 1010, 10)  # works the way I want
    # print(value_range)

    # index for finding the middle of the range
    index = int(len(value_range) / 2)  # works the way I want
    # print(value_range[index])

    start_time = time.time()
    # The line of code you want to measure
    results_all = omen.replicate(value_range[index])
    end_time = time.time()

    print(f"Execution time: {end_time - start_time:.6f} seconds")
