import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.sparse import diags
from scipy.sparse.linalg import splu
import matplotlib.animation as animation
# Constantes (unités naturelles hbar=1, m=1)
hbar = 1
m = 1

# Discrétisation spatiale
Nx = 10000  # beaucoup plus de points
L_affiche = 40  # demi-largeur affichée à l'écran
L_total = 40 * L_affiche  # domaine total très large
x = np.linspace(-L_total, L_total, Nx)
dx = x[1] - x[0]

# Potentiel : puits carré attractif
L = 5
V0 = -1.0  # profondeur du puits
V = np.zeros(Nx, dtype=complex)
V[(x > -L/2) & (x < L/2)] = V0

# Couche absorbante PML très large et très forte à droite (20% du domaine)
abs_width = int(0.2 * Nx)
abs_strength = 20.0  # absorption extrême
abs_profile = (np.linspace(0, 1, abs_width))**4
V[-abs_width:] = V[-abs_width:].real + 1j * abs_strength * abs_profile

# Paramètres temporels
dt = 0.5
Nt = 1000

def psi0(x, x0, sigma, k0):
    """Paquet d'onde gaussien complexe initial"""
    return (1/(sigma * np.sqrt(np.pi)))**0.5 * np.exp(-(x - x0)**2/(2 * sigma**2)) * np.exp(1j * k0 * x)

def normalize(psi):
    norm = np.sqrt(np.sum(np.abs(psi)**2)*dx)
    return psi / norm

def construct_operators(V, dt, dx, Nx):
    r = 1j * hbar * dt / (2 * m * dx**2)
    main_diag = 1 + 2*r + 1j * dt / (2 * hbar) * V
    off_diag = -r * np.ones(Nx-1)
    A = diags([off_diag, main_diag, off_diag], [-1,0,1], format='csc')

    main_diag_B = 1 - 2*r - 1j * dt / (2 * hbar) * V
    B = diags([r * np.ones(Nx-1), main_diag_B, r * np.ones(Nx-1)], [-1,0,1], format='csc')

    lu = splu(A)
    return B, lu

def step_CN(psi, B, lu):
    b = B.dot(psi)
    return lu.solve(b)

# Initial conditions
x0 = -20.0
sigma = 20
k0_init = 0.5

B, lu = construct_operators(V, dt, dx, Nx)
psi = normalize(psi0(x, x0, sigma, k0_init))

# --- Setup plot ---

fig, ax = plt.subplots(figsize=(10,5))
plt.subplots_adjust(bottom=0.25)

# Affichage centré sur la zone d'intérêt
mask_affiche = (x > -L_affiche) & (x < L_affiche)
# Affichage du puits de potentiel en gris
ax.fill_between(x[mask_affiche], 0, 0.5, where=(np.abs(x[mask_affiche]) < L/2), color='lightgrey', alpha=0.5, label='Puits')
line, = ax.plot(x[mask_affiche], np.abs(psi[mask_affiche])**2, lw=2)
ax.plot(x[mask_affiche], V[mask_affiche].real / abs(V0) * 0.3, 'r--', label='Potentiel (échelle)')
ax.set_xlim(-L_affiche, L_affiche)
ax.set_ylim(0, 0.5)
ax.set_xlabel('x')
ax.set_ylabel(r'$|\psi(x,t)|^2$')
title = ax.set_title(f'Propagation paquet d\'onde, k0 = {k0_init:.2f}')

# Slider k0
ax_k0 = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider_k0 = Slider(ax_k0, 'Impulsion k0', 0.1, 4.0, valinit=k0_init, valstep=0.01)

# Bouton reset
ax_button = plt.axes([0.8, 0.025, 0.1, 0.04])
button_reset = Button(ax_button, 'Reset', color='lightblue', hovercolor='0.97')

# Variables de contrôle animation
running = [True]
frame = [0]

def reset(event):
    frame[0] = 0
    global psi, B, lu
    psi = normalize(psi0(x, x0, sigma, slider_k0.val))
    B, lu = construct_operators(V, dt, dx, Nx)
    title.set_text(f'Propagation paquet d\'onde, k0 = {slider_k0.val:.2f}')

button_reset.on_clicked(reset)

def update_k0(val):
    reset(None)
slider_k0.on_changed(update_k0)

def update(frame_num):
    global psi
    if running[0]:
        psi = step_CN(psi, B, lu)
        psi = normalize(psi)
        line.set_ydata(np.abs(psi[mask_affiche])**2)
        title.set_text(f'Propagation paquet d\'onde, k0 = {slider_k0.val:.2f}, t = {frame_num*dt:.3f}')
    return line, title

ani = animation.FuncAnimation(fig, update, frames=Nt, interval=20, blit=False)

plt.show()

# --- Calcul et tracé du coefficient de transmission en fonction de l'énergie ---
def transmission_vs_energy(E_list, x, V, x0, sigma, hbar, m, dx, dt, Nt):
    T_list = []
    Nx = len(x)
    a = np.max(np.abs(x[V.real < 0])) if np.any(V.real < 0) else 0
    Nt_scan = 400  # Limite la durée de propagation pour chaque E
    for E in E_list:
        p0 = np.sqrt(2 * m * E)
        psi0_ = (1/np.sqrt(np.sqrt(np.pi)*sigma)) * np.exp(-(x - x0)**2/(2*sigma**2)) * np.exp(1j * p0 * x / hbar)
        psi0_ /= np.sqrt(np.sum(np.abs(psi0_)**2) * dx)
        psi_ = psi0_.copy()
        # Matrices Crank-Nicolson
        alpha = 1j * hbar * dt / (2 * m * dx**2)
        beta = 1j * dt / (2 * hbar)
        main_diag = 1 + 2 * alpha + beta * V
        off_diag = -alpha * np.ones(Nx - 1)
        A = diags([main_diag, off_diag, off_diag], [0, -1, 1], format='csc')
        main_diag_B = 1 - 2 * alpha - beta * V
        B = diags([main_diag_B, alpha * np.ones(Nx - 1), alpha * np.ones(Nx - 1)], [0, -1, 1], format='csc')
        lu_A = splu(A)
        def crank_nicolson_step(psi):
            rhs = B.dot(psi)
            psi_next = lu_A.solve(rhs)
            return psi_next
        # Propagation
        for _ in range(Nt_scan):
            psi_ = crank_nicolson_step(psi_)
        # Transmission: probabilité à droite du puits (hors puits et hors zone absorbante)
        mask_trans = (x > (L/2 + 5)) & (x < (x[-abs_width] - 5))
        T = np.sum(np.abs(psi_[mask_trans])**2) * dx
        T_list.append(T)
    return np.array(T_list)

# Paramètres pour le balayage
Emin = 0.01 * abs(V0)
Emax = 10 * abs(V0)
n_points = 200
E_list = np.linspace(Emin, Emax, n_points)
T_list = transmission_vs_energy(E_list, x, V, x0, sigma, hbar, m, dx, dt, Nt)

plt.figure(figsize=(8, 5))
plt.plot(E_list / abs(V0), T_list, 'b-', label='Transmission')
plt.xlabel("Energie / |V0|")
plt.ylabel("Coefficient de transmission T(E)")
plt.title("Transmission en fonction de E/|V0| (effet Ramsauer–Townsend)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
