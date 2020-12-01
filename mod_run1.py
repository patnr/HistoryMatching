import numpy as np
from matplotlib import pyplot as plt

import simulator.plotting as plots
from random_fields import gaussian_fields
from simulator import ResSim, simulate
from tools import sigmoid

# from mpl_tools.misc import freshfig


model = ResSim(Lx=2, Ly=1, Nx=20, Ny=20)

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
surf = sigmoid(surf)
surf = surf.reshape(model.shape)
# Insert barrier
surf[:model.Nx//2, model.Ny//3] = 0.001

# Set permeabilities to surf.
# Alternative: set S0 to it.
model.Gridded.K = np.stack([surf, surf])

# Define obs operator
obs_inds = [model.xy2ind(x, y) for (x, y, _) in model.producers]
def obs(saturation):
    return [saturation[i] for i in obs_inds]
obs.length = len(obs_inds)

# Simulate
S0 = np.zeros(model.M)
nTime = 28
saturation, production = simulate(model.step, nTime, S0, 0.025, obs)

# Plot IC
# fig, (ax1, ax2) = freshfig(47, figsize=(8, 4), ncols=2)
# contours = plots.field(model, ax1, surf)
# # fig.colorbar(contours)
# ax2.hist(surf.ravel())

# Animation
plots.COORD_TYPE = "index"
ani = plots.animate1(model, saturation, production)
plt.pause(.1)
