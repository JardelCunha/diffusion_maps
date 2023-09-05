import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class DiffusionMaps:
    def __init__(self, epsilon, alpha, dimension):
        self.epsilon = epsilon
        self.alpha = alpha
        self.dimension = dimension
        
    def calculate_distance_matrix(self, X):
        """
        Calculates the distance matrix between points in X.

        Parameters:
            X (numpy.ndarray): The input array of shape (num_points, num_features) containing the points.

        Returns:
            numpy.ndarray: The distance matrix of shape (num_points, num_points) where each element 
                           represents the distance between two points in X.
        """
        num_points = X.shape[0]
        distance_matrix = np.zeros((num_points, num_points))

        for i in range(num_points):
            for j in range(num_points):
                distance_matrix[i, j] = np.linalg.norm(X[i] - X[j])

        return distance_matrix

    def fit_transform(self, X):
        """
        Fits and transforms the input data using the spectral embedding algorithm.
        
        Parameters:
            X (array-like): Input data of shape (n_samples, n_features).
        
        Returns:
            Ut (array-like): Transformed data of shape (n_samples, dimension).
            None: If an error occurs during the calculation.
            str: Error message if an error occurs.
        """
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
            Ut = U_normalized

            return Ut

        except ValueError as e:
            return None, str(e)

def main():
    """
    Load the s_curve dataset from scikit-learn.
    
    Args:
        None
        
    Returns:
        None
    
    Comments:
        - The dataset is loaded using the make_swiss_roll function.
        - The number of samples is set to 1500.
        - The random_state is set to 42.
        - The features are stored in X.
        - The colors are stored in color.
        
    - The function creates two sets of parameters for the Diffusion Maps algorithm.
    - The parameters for the first configuration are:
        - epsilon = 5
        - alpha = 1
        - dimension = 3
    - The parameters for the second configuration are:
        - epsilon_new = 3.5
        - alpha_new = 1
        
    - Two instances of the DiffusionMaps class are created.
    - The first instance uses the parameters of the first configuration.
    - The second instance uses the parameters of the second configuration.
    
    - The Diffusion Maps algorithm is applied to both configurations.
    - The transformed data for the first configuration is stored in Ut.
    - The transformed data for the second configuration is stored in Ut_new.
    
    - The original data is plotted in a 3D scatter plot.
    - The transformed data for the first configuration is plotted in a 2D scatter plot.
    - The transformed data for the second configuration is plotted in a 2D scatter plot.
    """
    # Carregar o conjunto de dados s_curve do scikit-learn
    X, color = datasets.make_swiss_roll(n_samples=1500, random_state=42)

    # Parâmetros para as duas configurações do Diffusion Maps
    epsilon = 5
    alpha = 0.45
    dimension = 3

    epsilon_new = 12 # Nova configuração
    alpha_new = 1 # Nova configuração

    # Criar instâncias das classes DiffusionMaps para as duas configurações
    dm = DiffusionMaps(epsilon, alpha, dimension)
    dm_new = DiffusionMaps(epsilon_new, alpha_new, dimension)

    # Aplicar o Diffusion Maps para ambas as configurações
    Ut = dm.fit_transform(X)
    Ut_new = dm_new.fit_transform(X)

    cmap = plt.cm.Spectral
    
    # Plot dos dados originais em 3D
    fig = plt.figure(figsize=(18, 5))
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=cmap)
    ax1.set_title('Dados Originais')

    # Plot dos dados transformados em 2D (Configuração 1)
    ax2 = fig.add_subplot(132)
    ax2.scatter(Ut[:, 1], Ut[:, 2], c=color, cmap=cmap)
    ax2.set_title('Dados Transformados (Configuração 1)')
    ax2.set_xlabel(f'Epsilon = {epsilon}    Alpha = {alpha}')

    # Plot dos dados transformados em 2D (Configuração 2)
    ax3 = fig.add_subplot(133)
    ax3.scatter(Ut_new[:, 1], Ut_new[:, 2], c=color, cmap=cmap)
    ax3.set_title('Dados Transformados (Configuração 2)')
    ax3.set_xlabel(f'Epsilon = {epsilon_new}    Alpha = {alpha_new}')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()