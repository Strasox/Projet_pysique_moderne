import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math

# =========================================================================
# PARTIE 1 : SOLUTION ANALYTIQUE (COEFFICIENT DE TRANSMISSION)
# =========================================================================

# Paramètres physiques
V0 = -4000  # Profondeur du puits (en unités naturelles)
L = 0.1     # Largeur du puits
hbar = 1    # Constante de Planck réduite
m = 1       # Masse de la particule

def transmission_coefficient(E_ratio):
    """Calcule le coefficient de transmission théorique pour un puits carré."""
    E = E_ratio * abs(V0)
    if E <= 0:
        return 0.0
    k = np.sqrt(2*m*E) / hbar  # Vecteur d'onde à l'extérieur
    q = np.sqrt(2*m*(E - V0)) / hbar  # Vecteur d'onde à l'intérieur
    
    denominator = 1 + ((k**2 - q**2)**2 / (4 * k**2 * q**2)) * np.sin(q * L)**2
    return 1 / denominator

# Plage étendue d'énergies (E/V0 de 0.01 à 5)
E_ratios = np.linspace(0.01, 5, 1000)
T_theory = np.array([transmission_coefficient(er) for er in E_ratios])

# Identification des pics de transmission
peaks = []
for i in range(1, len(T_theory)-1):
    if T_theory[i] > T_theory[i-1] and T_theory[i] > T_theory[i+1] and T_theory[i] > 0.95:
        peaks.append(E_ratios[i])

# Tracé de la transmission théorique
plt.figure(figsize=(12, 6))
plt.plot(E_ratios, T_theory, 'b-', label="Théorie")
plt.xlabel(r"$E/|V_0|$", fontsize=14)
plt.ylabel("Coefficient de transmission $T$", fontsize=14)
plt.title("Effet Ramsauer-Townsend : Transmission théorique", fontsize=16)
plt.grid(True, alpha=0.3)

# Marquage des pics de résonance
for peak in peaks:
    plt.axvline(x=peak, color='r', linestyle='--', alpha=0.5)
    plt.text(peak, 0.5, f"{peak:.2f}", color='r', ha='center')

plt.ylim(0, 1.1)
plt.tight_layout()
plt.show()

# =========================================================================
# PARTIE 2 : SIMULATION NUMÉRIQUE (PAQUET D'ONDE)
# =========================================================================

# Paramètres de simulation (ajustés pour la performance)
dx = 0.002   # Pas spatial plus grand pour accélérer
nx = int(2 / dx)
dt = 5e-7    # Pas temporel plus grand
nt = 50000   # Nombre réduit d'itérations
save_every = 1000

def simulate_wave_packet(e_ratio, plot_animation=False):
    """Simule la propagation d'un paquet d'onde et calcule T."""
    E = e_ratio * abs(V0)
    x = np.linspace(0, (nx-1)*dx, nx)
    
    # Potentiel (décalé pour éviter les effets de bord)
    x0_pot = 0.8
    x1_pot = x0_pot + L
    V = np.zeros(nx)
    V[(x >= x0_pot) & (x <= x1_pot)] = V0
    
    # Paquet d'onde initial
    sigma = 0.05
    xc = 0.3
    k = math.sqrt(2*abs(E))
    A = 1/(sigma*np.pi)**0.25
    psi = A * np.exp(1j*k*x - (x-xc)**2/(2*sigma**2))
    
    re, im = np.real(psi), np.imag(psi)
    s = dt/dx**2
    
    densities = []
    for t in range(nt):
        if t % 2 == 0:
            im[1:-1] += s*(re[2:] + re[:-2] - 2*re[1:-1]) - dt*V[1:-1]*re[1:-1]
        else:
            re[1:-1] -= s*(im[2:] + im[:-2] - 2*im[1:-1]) - dt*V[1:-1]*im[1:-1]
        
        if t % save_every == 0:
            densities.append(re**2 + im**2)
    
    # Calcul de T sur la dernière frame
    last_density = densities[-1]
    transmitted = np.sum(last_density[x > x1_pot+0.1]) * dx
    total = np.sum(last_density) * dx
    T = transmitted / total
    
    if plot_animation:
        # Animation
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 10)
        line, = ax.plot([], [], 'b-')
        pot_line, = ax.plot(x, V/abs(V0), 'k-')
        
        def init():
            line.set_data([], [])
            return line,
        
        def update(i):
            line.set_data(x, densities[i])
            return line,
        
        ani = FuncAnimation(fig, update, frames=len(densities), init_func=init, 
                           blit=True, interval=50)
        plt.close()
        return T, ani
    else:
        return T, None

# Simulation pour une énergie de résonance
resonance_energy = peaks[0] if len(peaks) > 0 else 0.6
T_sim, ani = simulate_wave_packet(resonance_energy, plot_animation=True)

# Affichage de l'animation (dans Jupyter)
from IPython.display import HTML
HTML(ani.to_jshtml())

# Sauvegarde de l'animation
ani.save('ramsauer_animation.mp4', writer='ffmpeg', fps=20)

# =========================================================================
# PARTIE 3 : COMPARAISON THÉORIE/SIMULATION
# =========================================================================

# Calcul des points de simulation (moins nombreux pour gagner du temps)
sim_ratios = np.linspace(0.1, 5, 25)
T_simulations = []

for er in sim_ratios:
    T, _ = simulate_wave_packet(er)
    T_simulations.append(T)
    print(f"E/|V0| = {er:.2f}, T_sim = {T:.4f}")

# Tracé comparatif
plt.figure(figsize=(12, 6))
plt.plot(E_ratios, T_theory, 'b-', label="Théorie")
plt.plot(sim_ratios, T_simulations, 'ro', label="Simulation")
plt.xlabel(r"$E/|V_0|$", fontsize=14)
plt.ylabel("$T$", fontsize=14)
plt.title("Comparaison théorie-simulation", fontsize=16)
plt.legend()
plt.grid(True)

# Ajout des pics théoriques
for peak in peaks:
    plt.axvline(x=peak, color='r', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()