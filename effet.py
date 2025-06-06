import numpy as np
import matplotlib.pyplot as plt

hbar = 1.055e-34  #J.s
m = 9.11e-31      #kg
eV = 1.602e-19    #eV en joules


V0 = 1 * eV #J
a = 3e-9 #m

E_V0_ratios = np.linspace(0.01, 5, 1000) #valeur de début, fin, nombre de points à prendre dans cet intervalle
T_values = []

for ratio in E_V0_ratios:
    E = ratio * V0 #énergie en joules

    #Forme de T trouvée par l'étude analytique
    k2 = np.sqrt(2 * m * (E + V0)) / hbar #onde dans le puits
    T = 1 / (1 + (V0**2 * np.sin(k2 * a)**2) / (4 * E * (E + V0)))
    T_values.append(T)

# Tracé
plt.figure(figsize=(8,5))
plt.plot(E_V0_ratios, T_values, label="Puits de potentiel")
plt.xlabel(r"$E/V_0$")
plt.ylabel("Coefficient de transmission $T$")
plt.title("Evolution du coefficient de transmission T en fonction du rapport E/V0 avec a ="+str(a))
plt.grid(True)
plt.legend()
plt.show()
