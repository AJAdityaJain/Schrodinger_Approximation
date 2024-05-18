from scipy import sparse
from scipy.sparse.linalg import eigsh

import numpy as np
import matplotlib.pyplot as plt

q2_e0 = ((1.60217646e-2)**2)/ 8.854187812e-1

N = 256 #LOD of wavefunction
plot_size = 6 # Scale of plots
#Change potential accordingly
def get_potential(x, y, epsilon=1e-9):
    return 0*x#q2_e0 / (2 * np.pi * np.sqrt((x - 0.5)**2 + (y - 0.5)**2 + epsilon))


X, Y = np.mgrid[0:1:N*1j,0:1:N*1j]
V = get_potential(X,Y)

ones = np.ones([N])
grad = np.array([ones, -2*ones, ones])
diagonals = sparse.spdiags(grad, np.array([-1,0,1]), N, N)
Hamiltonian = (-1/2 * sparse.kronsum(diagonals,diagonals))+sparse.diags(V.reshape(N**2), (0))

print("EIGENING... Plz wait")
values , eigenvecs = eigsh(Hamiltonian, k=(plot_size*plot_size), which='SM')
fig, axs = plt.subplots(plot_size, plot_size, figsize=(12, 12))

ax = axs[0, 0]
for n in range(plot_size**2):
    ax = axs[n // plot_size, n % plot_size]
    c = ax.pcolormesh(X, Y, eigenvecs.T[n].reshape((N,N))**2, shading='auto')
    ax.set_title(f'n={n}')
    ax.axis('off')
    # ax.legend().set_visible(False)
    fig.colorbar(c, ax=ax)

plt.tight_layout(pad=0)
plt.axis('off')
plt.show()

