"""Illustrate model."""
import numpy as np
from matplotlib import pyplot as plt

import simulator.plotting as plots
from geostat import gaussian_fields
from simulator import ResSim, simulate

# from tools import sigmoid

# from mpl_tools.fig_layout import freshfig


model = ResSim(Lx=1, Ly=1, Nx=20, Ny=20)

# Relative coordinates
injectors = [[0.1, 0.0, 1.0], [0.9, 0.0, 1.0]]
producers = [[0.1, 0.7, 100.0], [0.9, 1.0, 1.0], [.5, .2, 1]]

# np.random.seed(1)
# injectors = rand(5,3)
# producers = rand(10,3)

model.init_Q(injectors, producers)

# Varying grid params
np.random.seed(3000)
surf = gaussian_fields(model.mesh(), 1)
surf = 0.5 + .2*surf
# surf = truncate_01(surf)
# surf = sigmoid(surf)
surf = surf.reshape(model.shape)
# Insert barrier
surf[:model.Nx//2, model.Ny//3] = 0.001

# Set permeabilities to surf.
# Alternative: set S0 to it.
model.Gridded.K = np.stack([surf, surf])  # type: ignore

# Define obs operator
obs_inds = [model.xy2ind(x, y) for (x, y, _) in model.producers]
def obs(saturation):  # noqa
    return [saturation[i] for i in obs_inds]
obs.length = len(obs_inds)  # noqa

# Simulate
S0 = np.zeros(model.M)  # type: ignore

# dt=0.025 was used in Matlab code with 64x64 (and 1x1),
# but I find that dt=0.1 works alright too.
# With 32x32 I find that dt=0.2 works fine.
# With 20x20 I find that dt=0.4 works fine.
T = 28*0.025
dt = 0.4
nTime = round(T/dt)
saturation, production = simulate(model.step, nTime, S0, dt, obs)

# Plot IC
# fig, (ax1, ax2) = freshfig(47, figsize=(8, 4), ncols=2)
# contours = plots.field(model, ax1, surf)
# # fig.colorbar(contours)
# ax2.hist(surf.ravel())

# Animation
plots.COORD_TYPE = "index"
ani = plots.dashboard(model, saturation, production, animate=False)
plt.pause(.1)
