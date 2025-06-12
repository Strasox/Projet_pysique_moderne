import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh  

dx = 0.001
L = 2.0
nx = int(L/dx)
x = np.linspace(0, L, nx)

V0 = -4000
V = np.zeros(nx)
V[(x >= 0.8) & (x <= 0.9)] = V0

# Construction du Hamiltonien (discrétisation du Laplacien)
main_diag = -2 * np.ones(nx)
off_diag = np.ones(nx - 1)
laplacien = (np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)) / dx**2

H = -0.5 * laplacien + np.diag(V)

num_etats = 10  # Nombre d’états stationnaires à extraire
energies, etats = eigh(H)

# Normalisation des états propres
etats_normalises = etats / np.sqrt(np.sum(np.abs(etats)**2) * dx)

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
