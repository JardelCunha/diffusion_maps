import numpy as np

class DiffusionMaps:
    def __init__(self, epsilon, alpha, dimension):
        self.epsilon = epsilon
        self.alpha = alpha
        self.dimension = dimension
        
    def calculate_distance_matrix(self, X):
        return np.linalg.norm(X[:, np.newaxis] - X, axis=2)

    def fit_transform(self, X):
        try:
            N = X.shape[0]

            # Step 1: Calculate the distance matrix D
            D = self.calculate_distance_matrix(X)

            # Step 2: Calculate the kernel matrix Ke
            Ke = np.exp(-D**2 / self.epsilon)

            # Step 3: Calculate the vector d
            d = Ke.sum(axis=1)

            # Step 4: Calculate the matrix Kea
            Kea = Ke / (np.outer(d, d)**self.alpha)

            # Step 5: Calculate the vector sqrtPi
            sqrtPi = np.sqrt(Kea.sum(axis=1))

            # Step 6: Calculate the matrix A
            A = Kea / np.outer(sqrtPi, sqrtPi)

            # Step 7: SVD decomposition
            U, _, _ = np.linalg.svd(A)

            # Step 8: Normalize the matrix U
            U_normalized = U / sqrtPi[:, np.newaxis]

            # Step 9: Select the first 'dimension' columns of U
            if self.dimension <= U_normalized.shape[1]:
                Ut = U_normalized[:, :self.dimension]
            else:
                raise ValueError("Dimension is too large.")

            return Ut, None

        except ValueError as e:
            return None, str(e)
'''
if __name__ == "__main__":
    np.random.seed(42)
    N = 100
    M = 10
    X = np.random.rand(N, M)

    epsilon = 0.1
    alpha = 0.5
    dimension = 2

    dm = DiffusionMaps(epsilon, alpha, dimension)
    Ut, error = dm.fit_transform(X)

    if error is None:
        print("Diffusion Maps transformation successful:")
        print(Ut)
    else:
        print(error)
'''