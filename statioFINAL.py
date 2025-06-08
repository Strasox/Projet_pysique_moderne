import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider

# Constantes (unités naturelles hbar=1, m=1)
hbar = 1
m = 1

# Potentiel
L = 5.0
V0 = -0.5  # profondeur du puits

# Domaine spatial
x_left = np.linspace(-40, -L/2, 400)
x_well = np.linspace(-L/2, L/2, 200)
x_right = np.linspace(L/2, 40, 400)
x = np.concatenate([x_left, x_well, x_right])

def calc_psi(E):
    """Calcule la fonction d'onde stationnaire psi pour une énergie E > 0"""
    if E <= 0:
        E = 0.01  # éviter les problèmes numériques

    k = np.sqrt(2 * m * E) / hbar
    q = np.sqrt(2 * m * (E - V0)) / hbar

    den = np.cos(q*L) - 1j*(k**2 + q**2)/(2*k*q)*np.sin(q*L)
    t_amp = np.exp(-1j*k*L) / den
    r_amp = 1j*(k**2 - q**2)/(2*k*q)*np.sin(q*L) / den

    psi_left = np.exp(1j*k*x_left) + r_amp * np.exp(-1j*k*x_left)
    psi_well = t_amp * (np.cos(q*(x_well + L/2)) - 1j*(k/q)*np.sin(q*(x_well + L/2)))
    psi_right = t_amp * np.exp(1j*k*x_right)

    return np.concatenate([psi_left, psi_well, psi_right]), E

# Initial energy
E0 = 1.0
psi, E = calc_psi(E0)

# Setup figure
fig, ax = plt.subplots(figsize=(10,5))
plt.subplots_adjust(bottom=0.25)

line, = ax.plot([], [], lw=2)
ax.axvline(-L/2, color='k', linestyle='--')
ax.axvline(L/2, color='k', linestyle='--')
ax.set_xlim(-40, 40)
ax.set_ylim(-2, 2)
ax.set_xlabel('x')
ax.set_ylabel(r'Re$(\psi(x,t))$')
title = ax.set_title(f'Animation onde stationnaire, E = {E0:.2f}')

# Slider axis
ax_E = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider_E = Slider(ax_E, 'Énergie E', 0.01, 3.0, valinit=E0, valstep=0.01)

# Animation function variables
time_vals = np.linspace(0, 20, 200)
frame_idx = [0]  # mutable container pour suivre l’indice dans animate

def init():
    line.set_data([], [])
    return line,

def animate(_):
    t = time_vals[frame_idx[0]]
    frame_idx[0] = (frame_idx[0] + 1) % len(time_vals)
    phase = np.exp(-1j * E * t / hbar)
    psi_t = psi * phase
    y = np.real(psi_t)
    line.set_data(x, y)
    title.set_text(f'Animation onde stationnaire, E = {E:.2f}, t = {t:.2f}')
    return line, title

def update(val):
    global psi, E, frame_idx
    E = slider_E.val
    psi, E = calc_psi(E)
    frame_idx[0] = 0  # reset animation time

slider_E.on_changed(update)

ani = animation.FuncAnimation(fig, animate, frames=len(time_vals),
                              init_func=init, blit=True, interval=50)

plt.show()

def transmission_analytique(E, L, V0, hbar=1, m=1):
    k = np.sqrt(2 * m * E) / hbar
    q = np.sqrt(2 * m * (E - V0)) / hbar
    den = np.cos(q*L) - 1j*(k**2 + q**2)/(2*k*q)*np.sin(q*L)
    T = np.abs(np.exp(-1j*k*L) / den)**2
    return T

# Tracé du coefficient de transmission en fonction de E/|V0|
E_vals = np.linspace(0.01, 10*abs(V0), 300)
T_vals = [transmission_analytique(E, L, V0, hbar, m) for E in E_vals]

plt.figure(figsize=(8,5))
plt.plot(E_vals/abs(V0), T_vals, 'b-', label='Transmission analytique')
plt.xlabel("Energie / |V0|")
plt.ylabel("Coefficient de transmission T(E)")
plt.title("Transmission stationnaire en fonction de E/|V0| (puits carré)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
