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
Nx = 10000
L_affiche = 40
L_total = 40 * L_affiche
x = np.linspace(-L_total, L_total, Nx)
dx = x[1] - x[0]

# Potentiel : puits carré attractif + couche absorbante (PML)
L = 5
V0 = -1.0
V = np.zeros(Nx, dtype=complex)
V[(x > -L/2) & (x < L/2)] = V0

abs_width = int(0.2 * Nx)
abs_strength = 20.0
abs_profile = (np.linspace(0, 1, abs_width))**4
V[-abs_width:] += 1j * abs_strength * abs_profile

# Paramètres temporels
dt = 0.5
Nt = 1000

def psi0(x, x0, sigma, k0):
    return (1/(sigma * np.sqrt(np.pi)))**0.5 * np.exp(-(x - x0)**2/(2 * sigma**2)) * np.exp(1j * k0 * x)

def normalize(psi):
    return psi / np.sqrt(np.sum(np.abs(psi)**2)*dx)

def construct_operators(V, dt, dx, Nx):
    r = 1j * hbar * dt / (2 * m * dx**2)
    main = 1 + 2*r + 1j * dt/(2*hbar) * V
    off  = -r * np.ones(Nx-1)
    A = diags([off, main, off], [-1,0,1], format='csc')
    mainB = 1 - 2*r - 1j * dt/(2*hbar) * V
    B = diags([r*np.ones(Nx-1), mainB, r*np.ones(Nx-1)], [-1,0,1], format='csc')
    return B, splu(A)

def step_CN(psi, B, lu):
    return lu.solve(B.dot(psi))

# Conditions initiales
x0, sigma = -20.0, 2
k0_init = 0.5
B, lu = construct_operators(V, dt, dx, Nx)
psi = normalize(psi0(x, x0, sigma, k0_init))

# Masque zone affichée
mask = (x > -L_affiche) & (x < L_affiche)

# --- Setup plot ---
fig, ax1 = plt.subplots(figsize=(10,5))
plt.subplots_adjust(bottom=0.25)

# Axe pour Re(psi)
ax1.set_xlim(-L_affiche, L_affiche)
ax1.set_ylim(-2, 2)
ax1.set_xlabel('x')
ax1.set_ylabel(r'Re$(\psi)$', color='C0')
line_psi, = ax1.plot([], [], lw=2, color='C0', label=r'Re$(\psi)$')
ax1.tick_params(axis='y', labelcolor='C0')

# Axe secondaire pour densité
ax2 = ax1.twinx()
ax2.set_ylim(0, 0.5)
ax2.set_ylabel(r'$|\psi|^2$', color='C1')
line_rho, = ax2.plot([], [], lw=2, linestyle='--', color='C1', label=r'$|\psi|^2$')
ax2.tick_params(axis='y', labelcolor='C1')

# Puits en gris, potentiel en échelle
ax1.fill_between(x[mask], 0, 0.5, where=(np.abs(x[mask])<L/2), color='lightgrey', alpha=0.5)
ax1.plot(x[mask], V[mask].real/abs(V0)*0.3, 'r--', label='Potentiel (échelle)')

title = ax1.set_title(f'Propagation paquet d\'onde, k0 = {k0_init:.2f}')

# Slider k0
ax_k0 = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider_k0 = Slider(ax_k0, 'Impulsion k0', 0.1, 4.0, valinit=k0_init, valstep=0.01)

# Bouton Reset
ax_button = plt.axes([0.8, 0.025, 0.1, 0.04])
button_reset = Button(ax_button, 'Reset', color='lightblue', hovercolor='0.97')

running = [True]
frame = [0]

def reset(event=None):
    frame[0] = 0
    global psi, B, lu
    psi = normalize(psi0(x, x0, sigma, slider_k0.val))
    B, lu = construct_operators(V, dt, dx, Nx)
    title.set_text(f'Propagation paquet d\'onde, k0 = {slider_k0.val:.2f}')

button_reset.on_clicked(reset)
slider_k0.on_changed(lambda val: reset())

def init():
    line_psi.set_data([], [])
    line_rho.set_data([], [])
    return line_psi, line_rho

def update(frame_num):
    global psi
    psi = step_CN(psi, B, lu)
    psi = normalize(psi)
    y_psi = np.real(psi)
    y_rho = np.abs(psi)**2
    line_psi.set_data(x[mask], y_psi[mask])
    line_rho.set_data(x[mask], y_rho[mask])
    title.set_text(f'k0 = {slider_k0.val:.2f}, t = {frame_num*dt:.2f}')
    return line_psi, line_rho

ani = animation.FuncAnimation(fig, update, frames=Nt, init_func=init,
                              blit=False, interval=20)

# Légende combinée
lines = [line_psi, line_rho]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right')

plt.show()

# --- Transmission vs énergie (scan E_list) ---
# (Votre fonction reste identique à l'original)
