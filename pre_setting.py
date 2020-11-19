import numpy as np 
from constants import *
from matplotlib.pyplot import * 
maxt = 90
## base state variables
theta = np.zeros(nz, ) # horizontal average ofpotential temperature
for i in range(1, len(theta)):
	height = dz * (-0.5 + i)
	if height <= z_tr:
		theta[i] = 300 + 43 * (height / z_tr) ** (1.25)
	else:
		theta[i] = theta_tr * np.exp(g * (height - z_tr) / (C_p * T_tr))
theta[0] = theta[1]
theta[-1] = theta[-2]

qvb = np.zeros(nz, ) # horizontal average of q_v
for i in range(1, len(qvb)):
	height = dz * (-0.5 + i)
	if height <= 4000:
		qvb[i] = 1.61e-2 - 3.375e-6 * height
	elif height <= 8000:
		qvb[i] = 2.6e-3 - 6.5e-7 * (height - 4000.)
qvb[0] = qvb[1]
qvb[-1] = qvb[-2]
#qvb = np.zeros(nz, ) #### reset qvb as 0 all to confirm surrounding is dry
qvb2d = np.tile(qvb, (nx, 1)) 

theta_v = theta * (1. + 0.61 * qvb) # horizontal average ofvirtual p.t.
pb = np.zeros(nz, ) # horizontal average of pressure
pib = np.zeros(nz, ) # horizontal average of non dimensional pressure
rhou = np.zeros(nz, ) # horizontal average of density in u level
rhow = np.zeros(nz, ) # horizontal average of density in w level
tb = np.zeros(nz, ) # horizontal average of temperature
qvsb = np.zeros(nz, ) # horizontal average of qvs
ub = np.zeros(nz, ) # horizontal average of u

## prognostic array (past(p), now, future(m) step)
thp, th, thm = np.zeros((nx, nz)), np.zeros((nx, nz)), np.zeros((nx, nz)) # perterbation of p.t. 
up, u, um = np.zeros((nx, nz)), np.zeros((nx, nz)), np.zeros((nx, nz)) # velocity u
wp, w, wm = np.zeros((nx, nz)), np.zeros((nx, nz)), np.zeros((nx, nz)) # velocity w
pip, pi, pim = np.zeros((nx, nz)), np.zeros((nx, nz)), np.zeros((nx, nz)) # perterbation of non dimensional pressure
qvp, qv, qvm = np.zeros((nx, nz)), np.zeros((nx, nz)), np.zeros((nx, nz)) # perterbation of water vapor
qcp, qc, qcm = np.zeros((nx, nz)), np.zeros((nx, nz)), np.zeros((nx, nz))# kg/ kg, the cloud mixing ratio
qrp, qr, qrm = np.zeros((nx, nz)), np.zeros((nx, nz)), np.zeros((nx, nz))# kg/ kg, the rain mixing ratio
## Temporal diffusion (Robert-Asselin filter)
thpd, thd = np.zeros((nx, nz)), np.zeros((nx, nz)) # perterbation of p.t. 
upd, ud = np.zeros((nx, nz)), np.zeros((nx, nz)) # velocity u
wpd, wd = np.zeros((nx, nz)), np.zeros((nx, nz)) # velocity w
pipd, pid = np.zeros((nx, nz)), np.zeros((nx, nz)) # perterbation of non dimensional pressure
qvpd, qvd = np.zeros((nx, nz)), np.zeros((nx, nz)) # perterbation of water vapor
qcpd, qcd = np.zeros((nx, nz)), np.zeros((nx, nz)) # cloud mixing ratio
qrpd, qrd = np.zeros((nx, nz)), np.zeros((nx, nz)) # rain mixing ratio
#3 Cloud physics
total_mr = np.zeros((nx, nz)) # kg / kg, 2D total mixing ratio
qvsp = np.zeros((nx, nz)) # kg / kg, 2D saturation mixing ratio
phi = np.zeros((nx, nz)) # supersaturation adjustment variable
A = np.zeros((nx, nz)) # Conversion of cloud water to rain water
B = np.zeros((nx, nz)) # Accretion of cloud water by rain water
C = np.zeros((nx, nz))
E = np.zeros((nx, nz))
V_term = np.zeros((nx, nz)) # terminal velocity of rain water
GAMMA = np.zeros((nx, nz))

def set_theta():
	"""set initial theta in grids and in level average"""
	for l in [xcnt]:
		for i in range(1, nx-1):
			for k in range(1, nz-1):
				#rad = np.sqrt((((k - .5) * dz - zcnt) / radz)**2 + (dx * (i - xmid) / radx)**2)
				rad = np.sqrt((((k - .5) * dz - zcnt) / radz)**2 + ((dx * (i - xmid) - l) / radx)**2)
				# .5 and .25 to adjust the center to (0, zcnt)
				if rad <= 1:
					th[i, k] += 0.5 * sigma * (np.cos(rad * np.pi) + 1 )
	## set boundary condition
	th[0, :] = th[nx-2, :] # colume boundary
	th[nx-1, :] = th[1, :] # colume boundary
	th[:, 0] = th[:, 1]
	th[:, nz-1] = th[:, nz-2]

def set_pib():
	"""set nondimensional pressure in level avarage"""
	for i in range(1, nz-1):
		if i == 1:
			pib[i] = (P_srf / P_0) ** (R_d / C_p)
		else: 
			pib[i] = pib[i-1] - dz * 2 * (g / C_p / (theta[i] + theta[i-1]))
	pib[0] = pib[1]
	pib[nz-1] = pib[nz-2]

def set_rho():
	"""calculate rhou and rhow"""
	mask = slice(1, nz-1)
	rhou[mask] = P_0 * pib[mask] ** (C_v / R_d) / (R_d * theta[mask])
	rhou[0], rhou[nz-1] = rhou[1], rhou[nz-2] # outer boundary = boundary

	rhow[mask] = (rhou[0:nz-2] + rhou[1:nz-1]) / 2
	rhow[0], rhow[nz-1] = rhow[1], rhow[nz-2] # outer boundary = boundary

def set_pi():
	"""calculate nondimensional perturbation """
	for i in range(1, nx-1):
		for k in reversed(range(1, nz-1)):
			tup = th[i, k+1] / theta[k+1]**2
			tdn = th[i, k] / theta[k]**2
			pi[i, k] = pi[i, k+1] - 0.5 * g / C_p * (tup + tdn) * dz

def set_ub():
	"""calculate the horizontal average of u each level, 
	the use is in the step 2: spatial diffusion"""
	for k in range(0, nz):
		ub[k] = np.mean(u[:, k])

def set_qv():
	"""let the perturbation of qv in the cell to be as environment
	the use is in the step 4: adding water vapor prog."""
	masks = th != 0
	qvsb2d = np.tile(qvsb, (nx, 1))
	qvb2d = np.tile(qvb, (nx, 1))
	qv[masks] = 0#qvsb2d[masks] - qvb2d[masks]

def set_bc(var_name, var):
	var[0, 1:nz-1] = var[nx-2, 1:nz-1] # column nx-2 copy to 0
	var[nx-1, 1:nz-1] = var[1, 1:nz-1] # column 1 copy to nx-1
	var[1:nx-1, 0] = var[1:nx-1, 1]
	var[1:nx-1, nz-1] = var[1:nx-1, nz-2]
	#if var_name == 'w':
	#	var[:, nz-1] = 0 # upper w = 0
	#	var[:, 0:2] = 0 # lower w = 0


## set up initial perturbation and their boundary
set_theta();
set_pib();
set_rho();
set_pi();

set_bc('th', th)
set_bc('pi', pi)
thp = th.copy()
pip = pi.copy()
## set_temporal_diffusion
thpd = thp.copy()
pipd = pip.copy()
upd = up.copy()
wpd = wp.copy()
thd = th.copy()
pid = pi.copy()
ud = u.copy()
wd = w.copy()


pb = P_0 * pib ** (C_p / R_d)
tb = theta * pib
qvsb = 380 / pb * np.exp(17.27 * (tb - 273.) / (tb-36))
for k in range(1, nz-1):
	qvsp[:, k] = 380 / (pb[k]) * np.exp(17.27 * ((thm[:, k] + theta[k]) * pib[k] - 273.) / ((thm[:, k] + theta[k]) * pib[k] - 36.))

set_qv();
#close qv = np.zeros((nx, nz))##close
qvp = qv.copy()
qvpd = qv.copy()
qvd = qv.copy()
#close qvb2d = np.zeros((nx, nz))##close
total_mr = qvb2d + qv

#figure(dpi=300);grid(True)
#plot(theta[1:-1], z_cor / 1000, 'r');xlabel('K');ylabel('height (km)');show()
#contourf(x_cor, z_cor, ((th)[1:nx-1, 1:nz-1].transpose()), levels=20);colorbar();show()
#imshow((total_mr)[1:nx-1, 1:nz-1][:, ::-1].transpose());colorbar();show()
#print(np.sum(qv!=0))