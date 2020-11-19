import numpy as np
### parameters ###
g = 9.81 # m / s^2
C_p = 1004 # J / (kg K)
C_v = 717 # J / (kg K)
P_0 = 1e5 # N / (m^2)
R_d = 287 # J / (kg K)
P_srf = 96500 # N / (m^2), surface pressure
cs = 50 # m / s, sound speed
Kx, Kz = 1500, 100 # m^2 / s, mixing coefficient
epsilon = 0.2 # temporal diffusion parameter
z_tr = 12000 # m, tropopause height
T_tr = 213 # k, temperature @ tropopause
theta_tr = 343 # K, potential temperature @ tropopause
L_v = 2.5e6 # J / kg, latent heat for vaporization
k1 = 1e-3 # parameter of A
k2 = 2.2 # s-1
qc0 = 1e-3 # kg / kg, threshold value of cloud water
rho_l = 1e3 # kg / m^3, liquid water density
N_0 = 8e6 # m^(-4), parameter of Marshall Palmer
C_D = 0.4
# == time control == #
dt = 2
maxt = 30
nowt = 0
radx = 2500 # m, radii of 
radz = 4000 # m, the perturbation
sigma = 3 # K, perturbation parameter, max of amplitude @ center
zcnt = 3000 # m, height of perturbation
xcnt = 0 # m, location x1 of perturbation
nx = 83 # # of grids in x-direction
nz = 42 # # of grids in z-direction
xmid = int((nx-1)/2) # index of mid x

## length of interval
dx = 400.
dz = 400.

## coordinates
x_cor = np.array([(x - .5) * dx for x in range(1, nx-1)]) # (!)
x_cor = x_cor - max(x_cor/2) - 1/4 * dx # (!)

z_cor = np.array([(z - .5) * dz for z in range(1, nz-1)]) # (!)
xvisind = slice(1, nx-1)
zvisind = slice(1, nz-1)