import numpy as np 
import matplotlib
from matplotlib.pyplot import * 
import matplotlib.cm as cm
from constants import *
from pre_setting import *
from nclcmaps import nclcmap

##
rain_rate = []



# ===============Dynamic Process==================

def du(i, k):
	hori_adv = -0.25 / dx * ((u[i+1, k] + u[i, k])**2 - (u[i, k] + u[i-1, k])**2)

	verti_adv = -0.25 / (rhou[k] * dz) * (rhow[k+1] * (w[i, k+1] + w[i-1, k+1]) * (u[i, k+1] + u[i, k]) - rhow[k] * (w[i, k] + w[i-1, k]) * (u[i, k] + u[i, k-1]))

	pgf = - C_p * theta_v[k] / dx * (pi[i, k] - pi[i-1, k])

	sdx = Kx / (dx ** 2) * (u[i+1, k] + u[i-1, k] - 2 * u[i, k])

	sdz = Kz / (dz ** 2) * (u[i, k+1] - ub[k+1] + u[i, k-1] - ub[k-1] - 2 * (u[i, k] - ub[k]))

	return hori_adv + verti_adv + pgf + sdx + sdz 

def dw(i, k):
	hori_adv = -0.25 / dx * ((u[i+1, k] + u[i+1, k-1]) * (w[i+1, k] + w[i, k]) - (u[i, k] + u[i, k-1]) * (w[i, k] + w[i-1, k]))

	verti_adv = -0.25 / (rhow[k] * dz) * (rhou[k] * (w[i, k+1] + w[i, k]) ** 2 - rhou[k-1] * (w[i, k] + w[i, k-1]) ** 2)

	pgf = - C_p * 0.5 * (theta_v[k] + theta_v[k-1]) * (pi[i, k] - pi[i, k-1]) / dz

	bouyancy = g * 0.5 * (th[i, k] / theta[k] + th[i, k-1] / theta[k-1]) + 0.5 * (0.61 * (qv[i, k] + qv[i, k-1]) - (qc[i, k] + qc[i, k-1]) - (qr[i, k] + qr[i, k-1]))

	sdx = Kx / (dx ** 2) * (w[i+1, k] + w[i-1, k] - 2 * w[i, k])

	sdz = Kz / (dz ** 2) * (w[i, k+1] + w[i, k-1] - 2 * w[i, k])

	return hori_adv + verti_adv + pgf + bouyancy + sdx + sdz 

def dth(i, k):
	hori_adv = -0.5 / dx * ((u[i+1, k]) * (th[i+1, k] + th[i, k]) - (u[i, k]) * (th[i, k] + th[i-1, k]))

	verti_adv = -0.5 / (rhou[k] * dz) * (rhow[k+1] * (w[i, k+1]) * (th[i, k+1] + th[i, k]) - rhow[k] * (w[i, k]) * (th[i, k] + th[i, k-1]))

	verti_adv_mean = -0.5 / (dz * rhou[k]) * (rhow[k+1] * w[i, k+1] * (theta[k+1] - theta[k]) + rhow[k] * w[i, k] * (theta[k] - theta[k-1]))

	sdx = Kx / (dx ** 2) * (th[i+1, k] + th[i-1, k] - 2 * th[i, k])

	sdz = Kz / (dz ** 2) * (th[i, k+1] + th[i, k-1] - 2 * th[i, k])

	return hori_adv + verti_adv + verti_adv_mean + sdx + sdz

def dpi(i, k):
	first = 1 / dx * rhou[k] * theta_v[k] * (u[i+1, k] - u[i, k])

	second = 0.5 / dz * (rhow[k+1] * w[i, k+1] * (theta_v[k+1] + theta_v[k]) - rhow[k] * w[i, k] * (theta_v[k] + theta_v[k-1]))

	sdx = Kx / (dx ** 2) * (pi[i+1, k] + pi[i-1, k] - 2 * pi[i, k])

	sdz = Kz / (dz ** 2) * (pi[i, k+1] + pi[i, k-1] - 2 * pi[i, k])

	return -cs**2 / (rhou[k] * C_p * (theta_v[k] ** 2)) * (first + second) + sdx + sdz 

def dqv(i, k):
	hori_adv = -0.5 / dx * ((u[i+1, k]) * (qv[i+1, k] + qv[i, k]) - (u[i, k]) * (qv[i, k] + qv[i-1, k]))

	verti_adv = -0.5 / (rhou[k] * dz) * (rhow[k+1] * (w[i, k+1]) * (qv[i, k+1] + qv[i, k]) - rhow[k] * (w[i, k]) * (qv[i, k] + qv[i, k-1]))

	verti_adv_mean = -0.5 / (dz * rhou[k]) * (rhow[k+1] * w[i, k+1] * (qvb[k+1] - qvb[k]) + rhow[k] * w[i, k] * (qvb[k] - qvb[k-1]))

	sdx = Kx / (dx ** 2) * (qv[i+1, k] + qv[i-1, k] - 2 * qv[i, k])

	sdz = Kz / (dz ** 2) * (qv[i, k+1] + qv[i, k-1] - 2 * qv[i, k])

	return hori_adv + verti_adv + verti_adv_mean + sdx + sdz 

def dqc(i, k):
	hori_adv = -0.5 / dx * ((u[i+1, k]) * (qc[i+1, k] + qc[i, k]) - (u[i, k]) * (qc[i, k] + qc[i-1, k]))

	verti_adv = -0.5 / (rhou[k] * dz) * (rhow[k+1] * (w[i, k+1]) * (qc[i, k+1] + qc[i, k]) - rhow[k] * (w[i, k]) * (qc[i, k] + qc[i, k-1]))

	sdx = Kx / (dx ** 2) * (qc[i+1, k] + qc[i-1, k] - 2 * qc[i, k])

	sdz = Kz / (dz ** 2) * (qc[i, k+1] + qc[i, k-1] - 2 * qc[i, k])

	return hori_adv + verti_adv + sdx + sdz 

def dqr(i, k):
	hori_adv = -0.5 / dx * ((u[i+1, k]) * (qr[i+1, k] + qr[i, k]) - (u[i, k]) * (qr[i, k] + qr[i-1, k]))

	verti_adv = -0.5 / (rhou[k] * dz) * (rhow[k+1] * (w[i, k+1] - V_term[i, k+1]) * (qr[i, k+1] + qr[i, k]) - rhow[k] * (w[i, k] - V_term[i, k]) * (qr[i, k] + qr[i, k-1]))

	sdx = Kx / (dx ** 2) * (qr[i+1, k] + qr[i-1, k] - 2 * qr[i, k])

	sdz = Kz / (dz ** 2) * (qr[i, k+1] + qr[i, k-1] - 2 * qr[i, k])

	return hori_adv + verti_adv + sdx + sdz 

def positive_definite(var):
	while np.sum(var<0):
		pos_mask = var > 0 # total mixing ratio > 0
		pos_sum = np.sum(var[pos_mask])
		neg_mask = var < 0 # total mixing ratio < 0
		neg_sum = np.sum(var[neg_mask]) # redundant vapor added
		var[neg_mask] = 0 # To make negative value = 0
		var[pos_mask] = var[pos_mask] - abs(neg_sum) / pos_sum * var[pos_mask]
	if np.sum(var<0): print('Warning for negative')
	return var

def ani(time, var, var_name):
	figure(dpi=300)
	xvisind = slice(1, nx-1)
	zvisind = slice(1, nz-1)

	if var_name == 'th':
		v_min, v_max = -10, 10
	#elif var_name == 'pi':
	#	v_min, v_max = -0.003, 0.003
	elif var_name == 'u':
		v_min, v_max = -10, 10
	elif var_name == 'w':
		v_min, v_max = -10, 10
	elif var_name == 'qv':
		v_min, v_max = -0.002, 0.002
	elif var_name == 'qc':
		v_min, v_max = 0, 0.005
	elif var_name == 'qr':
		v_min, v_max = 0, 0.0001

	"""cloud"""
	if var_name == 'cloud':
		contourf(x_cor, z_cor[1:26], (qc - 1e-6)[1:-1, 1:26].transpose(), vmin=0, vmax=0.004, cmap=nclcmap('gsltod'), levels=5);colorbar();
		contour(x_cor, z_cor[1:26], (qr - 1e-7)[1:-1, 1:26].transpose(), colors='deepskyblue', linewidths=.7, levels=5, alpha=.7)
		#contour(x_cor, z_cor[1:26], (qc - 1e-6)[1:-1, 1:26].transpose(), colors='grey', vmin=0, vmax=0.001, linewidths=.75, levels=10)

	elif var_name == 'theta':
	#theta
		contourf(x_cor, z_cor, (th + np.tile(theta, (nx, 1)))[1:-1, 1:-1].transpose(), levels=20, vmin=294, vmax=420, cmap=nclcmap('temp_19lev'));colorbar();
		contour(x_cor, z_cor, (th + np.tile(theta, (nx, 1)))[1:-1, 1:-1].transpose(), colors='black', levels=20, linewidths=.5);

	elif var_name == 'mr':
	#mixing ratio
		contourf(x_cor, z_cor, (qv + qvb2d - 1e-6)[1:-1, 1:-1].transpose(), cmap=nclcmap('MPL_GnBu'), levels=15, vmin=0, vmax=0.021);colorbar();
		contour(x_cor, z_cor, (qv + qvb2d - 1e-6)[1:-1, 1:-1].transpose(), colors='black', levels=15, linewidths=.5);

	elif var_name == 'u' or var_name == 'w':
		contourf(x_cor, z_cor, var[1:-1, 1:-1].transpose(), cmap=nclcmap('MPL_viridis'), levels=10, vmin=v_min, vmax=v_max);
		colorbar();
		contour(x_cor, z_cor, var[1:-1, 1:-1].transpose(), colors='black', levels=10, linewidths=.5)

	elif var_name == 'qc' or var_name == 'qr':
		contourf(x_cor, z_cor, var[1:-1, 1:-1].transpose(), cmap=nclcmap('MPL_viridis'), levels=10, vmin=v_min, vmax=v_max);
		colorbar();
		contour(x_cor, z_cor, var[1:-1, 1:-1].transpose(), colors='white', levels=10, linewidths=.5)

	elif var_name == 'qv':
		contourf(x_cor, z_cor, var[1:-1, 1:-1].transpose(), cmap=nclcmap('MPL_viridis'), levels=10, vmin=v_min, vmax=v_max);
		colorbar();
		contour(x_cor, z_cor, var[1:-1, 1:-1].transpose(), colors='black', levels=10, linewidths=.5)

	else:
		m = contourf(x_cor, z_cor, 
			var[xvisind, zvisind].transpose(), cmap=nclcmap('temp_19lev'), vmin=v_min, vmax=v_max, levels=15)
		contour(x_cor, z_cor, 
			var[xvisind, zvisind].transpose(), colors='black', vmin=v_min, vmax=v_max, 
			linewidths=.5, levels=15)
		colorbar(m)

	# add wind vector to th fig.
	if var_name == 'th':
		scale = np.sqrt(u[xvisind, zvisind].transpose() ** 2 + w[xvisind, zvisind].transpose() ** 2)
		quiver(x_cor, z_cor, u[xvisind, zvisind].transpose()/scale, w[xvisind, zvisind].transpose()/scale, 
				width=0.002, headwidth=2)

	title('time:'+str(time))
	savefig(var_name+'_'+str(sigma)+'K_'+str(time)+'.jpg')

	#draw()
	#pause(0.01)

	#show()
	clf()
	print('save figure in t: '+str(time)+var_name)


maxt = 2002

time_count = 0

while nowt <= maxt - dt or rr > 0:
	print(nowt)
	if nowt % 50 == 0:
		var_dict = {'cloud':0}#{'th':th, 'u':u, 'w':w, 'qv':qv, 'qc':qc, 'qr':qr, 'theta':0, 'cloud':0, 'mr':0}
		#{'th':th, 'pi':pi, 'u':u, 'w':w, 'qv':qv, 'qc':qc, 'qr':qr}

		for var_name, var in var_dict.items():
			ani(nowt, var - 1e-7, var_name)

	rr = -1000* np.dot(qr[1:-1, 2], (w - V_term)[1:-1, 2])
	print(rr)
	rain_rate.append(rr)

	"""Terminal Speed"""
	GAMMA = np.zeros((nx, nz))
	for i in range(1, nx-1):
		for k in range(1, nz-1):
			if qr[i, k] > 0:
				GAMMA[i, k] = np.sqrt(np.sqrt(rho_l * N_0 * np.pi / rhou[k] / qr[i, k]))
				#V_term[i, k] = np.sqrt(4 / 3 / C_D) * np.sqrt(g * rho_l / rhow[k] * np.pi) / 6 * 6.5625 / np.sqrt(GAMMA[i, k])
				V_term[i, k] = np.sqrt(8 / 3 / C_D) * np.sqrt(g * rho_l / rhow[k] * np.pi) * 6.5625 / np.sqrt(GAMMA[i, k])
			else: V_term[i, k] = 0
	set_bc('V_term', V_term)

	"""update the all properties in future"""
	for i in range(1, nx-1):
		for k in range(1, nz-1):
			um[i, k] = upd[i, k] + 2 * dt * du(i, k);
			if k != 1:
				wm[i, k] = wpd[i, k] + 2 * dt * dw(i, k);
			else:
				wm[i, k] = 0;
			thm[i, k] = thpd[i, k] + 2 * dt * dth(i, k);
			pim[i, k] = pipd[i, k] + 2 * dt * dpi(i, k);
			qvm[i, k] = qvpd[i, k] + 2 * dt * dqv(i, k);
			qcm[i, k] = qcpd[i, k] + 2 * dt * dqc(i, k);
			qrm[i, k] = qrpd[i, k] + 2 * dt * dqr(i, k);

	"""Positive-definite (moisture) advection (step 4)"""
	total_mr = qvb2d + qvm
	#total_mr[(total_mr)<0] = 0
	total_mr = positive_definite(total_mr)
	qvm = total_mr - qvb2d
	#qcm[qcm<0] = 0
	qcm = positive_definite(qcm)
	#qrm[qrm<0] = 0
	qrm = positive_definite(qrm)


	"""set periodic boundary"""
	for var_name, var in {'th':thm, 'pi':pim, 'u':um, 'w':wm, 'qv':qvm, 'qc':qcm, 'qr':qrm}.items():
		set_bc(var_name, var)

	"""Cloud water process"""
	for i in range(1, nx-1):
		for k in range(1, nz-1):
			# make sure no negative term inside
			mrplus = qv[i, k] + qvb[k] if (qv[i, k] + qvb[k]) > 0 else 0
			qcplus = qc[i, k] if qc[i, k] > 0 else 0
			qrplus = qr[i, k] if qr[i, k] > 0 else 0

			"""Conversion of cloud water to rain water (A)"""
			if qcplus > qc0:
				A[i, k] = k1 * (qcplus - qc0)
				A[i, k] = min(A[i, k] * dt, qcplus)
				qcm[i, k] -= A[i, k]
				qrm[i, k] += A[i, k]

			"""Accretion  of rain water from cloud water (B)"""
			if qcplus or qrplus:
				B[i, k] = rhou[k] * k2 * qcplus * (qrplus ** (0.875))
				B[i, k] = min(B[i, k] * dt, qcplus)
				qrm[i, k] += B[i, k]
				qcm[i, k] -= B[i, k]

			"""Evaporation of rain water (E)"""
			if qrplus > 0:
				qvs = 380 / (pb[k]) * np.exp(17.27 * ((th[i, k] + theta[k]) * pib[k] - 273) / ((th[i, k] + theta[k]) * pib[k] - 36))
				C_vent = 1.6 + 30.39 * (rhou[k] * qrplus) ** (0.2046)
				diff_qv = (1 - (mrplus) / qvs) if (1 - (mrplus) / qvs) > 0 else 0
				E[i, k] = 1 / rhou[k] * (diff_qv * C_vent * ((rhou[k] * qrplus) ** 0.525)) / (2.03e4 + 9.584e6 / (pb[k] * qvs))
				E[i, k] = min(max(0, E[i, k] * dt), qrplus)
				qrm[i, k] -= E[i, k]
				qvm[i, k] += E[i, k]
				thm[i, k] -= L_v / (C_p * pib[k]) * E[i, k]

	"""Positive-definite (moisture) advection (step 4)"""
	total_mr = qvb2d + qvm
	#total_mr[(total_mr)<0] = 0
	total_mr = positive_definite(total_mr)
	qvm = total_mr - qvb2d
	#qcm[qcm<0] = 0
	qcm = positive_definite(qcm)
	#qrm[qrm<0] = 0
	qrm = positive_definite(qrm)
	total_mr = qvm + qvb2d


	# Check for supersaturation adjustment
	for i in range(1, nx-1):
		for k in range(1, nz-1):
			qvs = 380 / (pb[k]) * np.exp(17.27 * ((thm[i, k] + theta[k]) * pib[k] - 273.) / ((thm[i, k] + theta[k]) * pib[k] - 36))
			phi[i-1, k-1] = qvs * (17.27 * 237 * L_v / (C_p * ((thm[i, k] + theta[k]) * pib[k] - 36)**2))
			C[i-1, k-1] = abs((total_mr[i, k] - qvs) / (1 + phi[i-1, k-1])) # C > 0: oversaturated

			if total_mr[i, k] > qvs : # if oversaturated and qc is enough
				qvm[i, k] -= C[i-1, k-1]
				qcm[i, k] += C[i-1, k-1]
				thm[i, k] += L_v / (C_p * pib[k]) * C[i-1, k-1]

			elif total_mr[i, k] < qvs and qcm[i, k] > 0: # if qc exist to fill up and not saturated ##???
				temp = min(C[i-1, k-1], qcm[i, k])
				thm[i, k] -= L_v / (C_p * pib[k]) * temp
				qcm[i, k] -= temp
				qvm[i, k] += temp

	if np.sum((qvm+qvb2d)[1:-1, 1:-1]<0): print('error1'*10)
	if np.sum((qcm)[1:-1, 1:-1]<0): print('error2'*10)
	if np.sum((qrm)[1:-1, 1:-1]<0): print('error3'*10)

	# set periodic boundary
	for var_name, var in {'th':thm, 'pi':pim, 'u':um, 'w':wm, 'qv':qvm, 'qc':qcm, 'qr':qrm}.items():
		set_bc(var_name, var)

	# Temporal diffusion (Robert-Asselin filter)
	thd = th + epsilon / 2 * (thm + thpd - 2 * th)
	pid = pi + epsilon / 2 * (pim + pipd - 2 * pi)
	ud  = u + epsilon / 2 * (um + upd - 2 * u)
	wd  = w + epsilon / 2 * (wm + wpd - 2 * w)
	qvd = qv + epsilon / 2 * (qvm + qvpd - 2 * qv)
	qcd = qc + epsilon / 2 * (qcm + qcpd - 2 * qc)
	qrd = qr + epsilon / 2 * (qrm + qrpd - 2 * qr)

	set_ub()

	up = u.copy()
	u = um.copy()
	wp = w.copy()
	w = wm.copy()
	thp = th.copy()
	th = thm.copy()
	pip = pi.copy()
	pi = pim.copy()
	qvp = qv.copy()
	qv = qvm.copy()
	qcp = qc.copy()
	qc = qcm.copy()
	qrp = qr.copy()
	qr = qrm.copy()

	thpd = thd.copy()
	pipd = pid.copy()
	upd = ud.copy()
	wpd = wd.copy()
	qvpd = qvd.copy()
	qcpd = qcd.copy()
	qrpd = qrd.copy()

	time_count += 1
	nowt = dt * time_count

rain_rate = np.array(rain_rate)
np.save('rr_75.npy', rain_rate)
print('sucess')