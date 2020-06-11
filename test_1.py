import numpy as np
from simulation import simulate

def test_main():
    P,V,S = simulate(1,plotting=False)
    assert np.isclose(S[2900], 0.019031377764138553, atol=1e-14)

def test_main2():
    P,V,S = simulate(1,plotting=False)
    assert np.isclose(P[45,13], 0.3026998009061621, atol=1e-14)
