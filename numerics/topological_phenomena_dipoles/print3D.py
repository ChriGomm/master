import numpy as np
import matplotlib.pyplot as plt

eigenvals = np.load("numerics/3D_data/periodic_dipole_eigenvalues.npy")
kx, ky = np.load("numerics/3D_data/k_periodic_dipole.npy")
grid_size  = eigenvals.shape[0]
from matplotlib import cm

plt.style.use('_mpl-gallery')




fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=(8, 6))
# for i in range(4,eigenvals.shape[2]):
for i in range(4,6):
    # ax.plot_surface(kx[:grid_size//2,:], ky[:grid_size//2,:], eigenvals[:grid_size//2,:,i],color=cm.viridis(i/eigenvals.shape[2]), alpha=0.7)
    # for x in range(grid_size):
    #     for y in range(grid_size):
    ax.scatter(kx, ky, np.real(eigenvals[:,:,i]), color=cm.viridis(i/eigenvals.shape[2]), alpha=0.7,s=20)
ax.view_init(elev=5, azim=30)

plt.show()

# import matplotlib.pyplot as plt
# import numpy as np

# from matplotlib import cm
# from matplotlib.ticker import LinearLocator

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# # Make data.
# X = np.arange(-5, 5, 0.25)
# Y = np.arange(-5, 5, 0.25)
# X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)

# # Plot the surface.
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)

# # Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# # A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter('{x:.02f}')

# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)

# plt.show()