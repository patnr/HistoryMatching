import numpy as np
from main import main

def test_main():
    P,V,S = main(nSteps=1,plotting=False)
    assert np.isclose(S[20], 0.9471748473345845)

def test_main2():
    P,V,S = main(nSteps=1,plotting=False)
    assert np.isclose(P[20,20,0], -2.362583278529025)
