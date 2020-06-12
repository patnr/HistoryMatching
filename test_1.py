import numpy as np
from simulation import simulate,M

def test_main():
    S0 = np.zeros(M)
    saturation,production = simulate(1,S0,.025,dt_plot=None)
    assert np.isclose(saturation[0][82], 0.7283060464780774, atol=1e-14)
