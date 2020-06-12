import numpy as np
from simulation import simulate, S0

def test_main():
    saturation,production = simulate(1,S0,.025,dt_plot=None)
    assert np.isclose(saturation[-1][96], 0.355432939161119, atol=1e-14)
