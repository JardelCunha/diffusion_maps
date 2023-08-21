import matplotlib.pyplot as plt
from sklearn import datasets, manifold
from mpl_toolkits.mplot3d import Axes3D
from DiffusionMaps import *

X, color = datasets.make_swiss_roll(n_samples=1500, random_state=42)

# Parâmetros para o Diffusion Maps
epsilon = 2
alpha = 0.8
dimension = 3
t = 1

# Aplicar o Diffusion Maps
dm = DiffusionMaps(epsilon, alpha, dimension)
Ut, error = dm.fit_transform(X)

# Plot dos dados originais em 3D
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
ax1.set_title('Dados Originais')

print(Ut.shape)
# Plot dos dados transformados em 2D
ax2 = fig.add_subplot(122)
#descartando a primeira diemsão, pois não é relevante para visualização
ax2.scatter(Ut[:, 1], Ut[:, 2], c=color, cmap=plt.cm.Spectral)
ax2.set_title('Dados Transformados')

plt.tight_layout()
plt.show()