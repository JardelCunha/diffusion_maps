import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import DiffusionMaps

def main():
    # Carregar o conjunto de dados s_curve do scikit-learn
    X, color = datasets.make_s_curve(n_samples=1000, random_state=42)

    # Parâmetros para as duas configurações do Diffusion Maps
    epsilon = 5
    alpha = 1
    dimension = 3

    epsilon_new = 0.3  # Nova configuração
    alpha_new = 1  # Nova configuração

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