import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
from matplotlib.pyplot import *

def etats(x,V,nE,tol=0):
    N = V.size
    if x.size!=N:
        raise Exception("tailles de x et V incompatibles")
    alpha = 1.0/(x[1]-x[0])**2
    H = sparse.diags([-alpha/2,V+alpha,-alpha/2],[-1,0,1],shape=(N,N))
    E,phi = linalg.eigs(H,nE,which='SM',tol=tol)
    return np.real(E),phi

x = np.linspace(-1.0,1.0,500)

V0 = -100
V = np.zeros(x.size)
V[(x >= -0.5) & (x <= 0.5)] = V0

E,phi = etats(x,V,10)
figure(figsize=(12,8))
i=0
for i in range(E.size):
    plot(x,np.square(np.absolute(phi[:,i])),label="E%d=%f"%(i,E[i]))
xlabel("x")
ylabel("|psi|^2")
grid()
legend(loc='upper right')

nE = 20
E,phi = etats(x,V,nE)
figure()
plot(range(1,nE+1),E,'o')
xlabel("n")
ylabel("E")
grid()


show()