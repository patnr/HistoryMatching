import numpy as np
from simulation import simulate, M, injectors, xy2i

xy = injectors[0,:2]
ind = xy2i(*xy)


# def test_main():
    # S0 = np.zeros(M)
    # saturation,production = simulate(1,S0,.025,dt_plot=None)
    # assert np.isclose(saturation[0][ind+1], 0.6788724415416183, atol=1e-14)
