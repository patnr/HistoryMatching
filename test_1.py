import numpy as np
from main import main

Dx = 1
Dy = 1
Dz = 1
Grid = dict(
    Nx = 64,
    Ny = 64,
    Nz = 1 ,
)
Grid['hx'] = Dx/Grid['Nx'] ; # Dimension in x-direction
Grid['hy'] = Dy/Grid['Ny'] ; # Dimension in y-direction
Grid['hz'] = Dz/Grid['Nz'] ; # Dimension in z-direction

##
N=Grid['Nx']*Grid['Ny']; # Total number of grid blocks


Grid['V']=Grid['hx']*Grid['hy']*Grid['hz']; # Cell volumes
# TODO: matlab actually squeezes last ax (coz its 1):
Grid['K']=np.ones((3,Grid['Nx'],Grid['Ny'],Grid['Nz'])); # Unit permeability
Grid['por'] =np.ones((Grid['Nx'],Grid['Ny'],Grid['Nz'])); # Unit porosity

##
Q=np.zeros(N); Q[0]=1; Q[-1] = -1; Q = Q[:,None]; # Production/injection
Fluid = {}
Fluid['vw']=1.0; Fluid['vo']=1.0; # Viscosities
Fluid['swc']=0.0; Fluid['sor']=0.0; # Irreducible saturations

def test_main():
    P,V,S = main(1)
    assert np.isclose(S[20], 0.9471748473345845)
def test_main2():
    P,V,S = main(1)
    assert np.isclose(P[20,20,0], -2.362583278529025)
