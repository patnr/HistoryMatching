import numpy as np
from simulation import simulate

def test_main():
    P,V,S = simulate(1,plotting=False)
    assert np.isclose(S[45], 0.714336657591818, atol=1e-14)

def test_main2():
    P,V,S = simulate(1,plotting=False)
    assert np.isclose(P[41,0], 0.4203455218286327, atol=1e-14)
