import numpy as np
import matplotlib.pyplot as plt
import math

# Paramètres constants
dt = 1E-7
dx = 0.001
nx = int(1 / dx) * 2
nt = 60000
s = dt / dx ** 2
sigma = 0.05
A = 1 / (math.sqrt(sigma * math.sqrt(math.pi)))
xc = 0.4
v0 = -4000  # profondeur du puits
a, b = 0.8, 1.2  # limites du puits

# Grille d'espace
o = np.linspace(0, (nx - 1) * dx, nx)
V = np.zeros(nx)
V[(o >= a) & (o <= b)] = v0

# Liste des rapports E/|V0|
e_values = np.linspace(3, 6, 31)
T_values = []

for e in e_values:
    E = e * abs(v0)
    k = np.sqrt(2 * E)

    # Paquet initial
    cpt = A * np.exp(1j * k * o - ((o - xc) ** 2) / (2 * sigma ** 2))
    re = np.real(cpt)
    im = np.imag(cpt)
    densite = np.zeros((nt, nx))
    densite[0, :] = np.abs(cpt) ** 2
    b = np.zeros(nx)

    # Simulation temporelle
    for i in range(1, nt):
        if i % 2 != 0:
            b[1:-1] = im[1:-1]
            im[1:-1] = im[1:-1] + s * (re[2:] + re[:-2] - 2 * re[1:-1]) - 2 * dt * V[1:-1] * re[1:-1]
            densite[i, 1:-1] = re[1:-1]**2 + im[1:-1]*b[1:-1]
        else:
            re[1:-1] = re[1:-1] - s * (im[2:] + im[:-2] - 2 * im[1:-1]) + 2 * dt * V[1:-1] * im[1:-1]

    # Transmission : intégrale de la densité à droite du puits
    dens_finale = densite[-1, :]
    zone_transmise = o > (b + 0.05)
    proba_transmise = np.sum(dens_finale[zone_transmise]) * dx
    proba_totale = np.sum(dens_finale) * dx
    T = proba_transmise / proba_totale
    T_values.append(T)

    print(f"E/|V0| = {e:.2f}, T = {T:.4f}")

plt.plot(e_values, T_values)
plt.title("Effet Ramsauer-Townsend : Transmission en fonction de E/|V₀|")
plt.xlabel("E / |V₀|")
plt.ylabel("Coefficient de transmission T")
plt.grid(True)
plt.ylim(0.6, 1.05)
plt.xlim(3,6)
plt.show()