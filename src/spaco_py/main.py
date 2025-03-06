import numpy as np
from spaco_py.SpaCoObject import SPACO


def main():
    random_matrix = np.random.rand(5, 5)
    neighbor_matrix = np.random.rand(5, 5)
    spatial_component_class = SPACO(random_matrix, neighbor_matrix)
    scale_center_data = spatial_component_class.__preprocess(random_matrix)
    print(scale_center_data)


if __name__ == "__main__":
    main()
