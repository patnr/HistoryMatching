import numpy as np
from simulation import simulate, injectors, xy2i, xy2sub

xy = injectors[:2,0]
def test_main():
    P,V,S = simulate(1,plotting=False)
    assert np.isclose(S[xy2i(*xy)], 0.9540751770156238, atol=1e-14)

def test_main2():
    P,V,S = simulate(1,plotting=False)
    assert np.isclose(P[xy2sub(*xy)], 0.09942313923592554, atol=1e-14)
