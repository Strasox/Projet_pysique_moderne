import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh  # Solveur pour matrices Hermitiennes

dx = 0.001
L = 2.0
nx = int(L/dx)
x = np.linspace(0, L, nx)

V0 = -4000
V = np.zeros(nx)
V[(x >= 0.8) & (x <= 0.9)] = V0

# Construction du Hamiltonien (discrétisation du Laplacien)
diag = np.ones(nx) * (2 / dx**2)
off_diag = np.ones(nx - 1) * (-1 / dx**2)

H = np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
H += np.diag(V)

num_etats = 5  # Nombre d’états stationnaires à extraire
energies, etats = eigh(H, subset_by_index=(0, num_etats-1))

# Normalisation des états propres
etats_normalises = etats / np.sqrt(np.sum(np.abs(etats)**2, axis=0) * dx)

# Affichage des états stationnaires
plt.figure(figsize=(10, 6))
for i in range(num_etats):
    plt.plot(x, etats_normalises[:, i]**2 + energies[i], label=f"État {i}, E = {energies[i]:.2f}")

plt.plot(x, V, color='k', linestyle='--', label="Potentiel V(x)")
plt.xlabel("x")
plt.ylabel("ψ(x)^2 + E")
plt.title("États stationnaires et densités de probabilité")
plt.legend()
plt.grid()
plt.show()
