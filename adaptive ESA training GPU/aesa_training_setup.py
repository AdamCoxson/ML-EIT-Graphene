import numpy as np
import greit_rec_training_set as train
import cupy as cp
from pyeit.eit.fem import Forward as FEM_normal
from pyeit.eit.fem_forallmeas import Forward
from pyeit.eit.fem_for_given_meas import Forward as FEM
from pyeit.eit.utils import eit_scan_lines
from meshing import mesh

from time import time
import h5py as h5
import sys
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from measurement_optimizer import *

def saveJacobian(n_el=20, n_per_el=3):
	# number electrodes
	el_pos = np.arange(n_el * n_per_el)
	# create an object with the meshing characteristics to initialise a Forward object
	mesh_obj = mesh(n_el)

	fwd = Forward(mesh_obj, el_pos, n_el)

	ex_mat = train.generateExMat(ne=n_el)

	f, meas, new_ind = fwd.solve_eit(ex_mat=ex_mat, perm=fwd.tri_perm)
	#print(f)
	ind = np.arange(len(meas))

	np.random.shuffle(ind)

	pde_result = train.namedtuple("pde_result", ['jac', 'v', 'b_matrix'])

	f = pde_result(jac=f.jac[ind],
				   v=f.v[ind],
				   b_matrix=f.b_matrix[ind])
	meas = meas[ind]
	new_ind = new_ind[ind]
	h = h5.File('forward.h5', 'w')

	try:
		h.create_dataset('jac', data=f.jac)
		h.create_dataset('v', data=f.v)
		h.create_dataset('b', data=f.b_matrix)
		h.create_dataset('meas', data=meas)
		h.create_dataset('new_ind', data=new_ind)
		h.create_dataset('p', data=mesh_obj['node'])
		h.create_dataset('t', data=mesh_obj['element'])
	except:
		TypeError('Error with saving files!')

	h.close()

def solveGREIT(n_el=20, n_pix=64, a=2.):
	# number electrodes because these are their indices in pts array (due to pyEIT meshing)
	el_pos = np.arange(n_el)
	# create an object with the meshing to initialise a Forward object
	mesh_obj = mesh(n_el)
	fwd = Forward(mesh_obj, el_pos)
	ex_mat = train.generateExMat(ne=n_el)
	f = fwd.solve_eit(ex_mat=ex_mat, perm=fwd.tri_perm)
	# generate anomalies in a random manner to simulate real samples
	anomaly = train.generate_anoms(a, a)
	# generate new meshing with anomaly
	new_mesh = train.set_perm(mesh_obj, anomaly=anomaly, background=None)
	# solve forward problem with anomaly
	f_anom = fwd.solve_eit(ex_mat=ex_mat, perm=new_mesh['perm'])
	greit = train.greit.GREIT(mesh_obj, el_pos, f=f, ex_mat=ex_mat, step=None)
	greit.setup(p=0.5, lamb=0.1, n=n_pix)
	rel_perm = greit.solve(f_anom.v, f.v)
	rel_perm = rel_perm.reshape((n_pix, n_pix)) # from (n_pix, n_pix, 1) to (n_pix, n_pix)

	return rel_perm

def lossL2(im, true):
	return np.sum((im - true)**2)

def testAlgorithm(fileJac, n=4, a=2., ne=20, pert=0.5, cutoff=0.97, p_influence=-10., p_rec=10.):
	file = h5.File(fileJac, 'r')
	meas = file['meas'][()]
	new_ind = file['new_ind'][()]
	p = file['p'][()]
	t = file['t'][()]
	'''jac = file['jac'][()]
	b = file['b'][()]
	v = file['v'][()]'''
	file.close()
	#make anomaly
	anomaly = None
	while anomaly is None:
		anomaly = train.generate_anoms(a, a)
	print(anomaly)
	index_stand = ((np.absolute(meas[:,1] - meas[:,0]) == 10))# + (np.absolute(meas[:,1] - meas[:,0]) == 19))
	index_stand *= ((np.absolute(meas[:,3] - meas[:,2]) == 1) + (np.absolute(meas[:,3] - meas[:,2]) == 19))

	ex_mat_all = train.generateExMat(ne=ne)
	volt_mat_all = train.generateExMat(ne=ne)
	index_1 = ((np.absolute(volt_mat_all[:,1] - volt_mat_all[:,0]) == 1) + (np.absolute(volt_mat_all[:,1] - volt_mat_all[:,0]) == 19))
	volt_mat_1 = volt_mat_all[index_1]
	volt_mat_1 = volt_mat_1[np.argsort(volt_mat_1[:, 1], axis=0)]
	volt_mat_1 = volt_mat_1[np.argsort(volt_mat_1[:, 0]%5, axis=0)]
	#print(np.sum(index_stand))
	meas_11 = meas[index_stand]
	ind = np.argsort(meas_11[:, 1], axis=0)
	meas_11 = meas_11[ind]

	ordered = meas_11[:10]
	suggested = ordered[:]
	counter = np.zeros(190)
	print(ordered)
	numS = []
	numO = []
	lossS = []
	lossO = []
	for i in range(int((len(meas_11)-10)//n)):
		ex_pred, recSugg, trSugg, numSugg, total_map = findNextPair(fileJac, ne, anomaly=anomaly, meas=suggested, a=2., npix=64, pert=pert, p_influence=p_influence, p_rec=p_rec)

		recOrd, _, _, _, trOrd, numOrd = simulateMeasurements(fileJac, anomaly=anomaly, measurements=ordered, n_el=ne, n_pix=64, a=a)

		lossSugg = lossL2(recSugg, trSugg)
		lossOrd = lossL2(recOrd, trOrd)

		numS.append(numSugg)
		numO.append(numOrd)
		lossS.append(lossSugg)
		lossO.append(lossOrd)

		print('Reconsructing with', 10 + i * n, 'measurements:')
		print('Loss Optimised =', lossSugg)
		print('Loss Ordinary =', lossOrd)

		rem_r = (np.sum((volt_mat_1 != ex_pred[0]) + (volt_mat_1 != ex_pred[1]), axis=1)).astype(bool)
		volt_mat = volt_mat_1[rem_r]
		j = 0
		while True:
			loc = (np.sum((ex_mat_all == ex_pred[None, j]), axis=1) == 2).astype(bool)
			if counter[loc] == 0 or counter[loc] < (ne - 2) * (ne - 3)//(5*n):
				counter[loc] += 1
				ex_pred = ex_pred[j]
				new_volt_meas = findNextVoltagePair(ex_pred, fileJac, total_map, n, counter[loc], npix=64, cutoff=cutoff)
				break
			else:
				j += 1
		#x = volt_mat[int(n * (counter[loc] - 1)):int(n * counter[loc])]
		#x = np.hstack([np.tile(ex_pred, (n, 1)), x])
		x = np.empty((new_volt_meas.shape[0], 4))
		x[:, :2] = ex_pred
		x[:, 2:] = new_volt_meas
		#print(x)
		ordered = meas_11[:int(10 + n * (i+1))]
		suggested = np.concatenate((suggested, x), axis=0)

	
	fig1, ax1 = plt.subplots()
	ax1.plot(numS, lossS, 'rx')
	ax1.set_title('L2 Loss Function of Algorithm Selected Measurements')
	ax1.set_xlabel('Num. of Measurements')
	ax1.set_ylabel('L2 Loss Squared')

	fig2, ax2 = plt.subplots()
	ax2.plot(numO, lossO, 'bx')
	ax2.set_title('L2 Loss Function of a Commonly Used Measurement Technique')
	ax2.set_xlabel('Num. of Test Sample')
	ax2.set_ylabel('L2 Loss Squared')
	plt.show()
	parameterS = np.gradient(lossS) / np.gradient(numS)
	parameterS /= np.array(numS)**1.
	parameterS = np.mean(parameterS)

	parameterO = np.gradient(lossO) / np.gradient(numO)
	parameterO /= np.array(numO)**1.
	parameterO = np.mean(parameterO)
	return np.array(lossS), np.array(numS), np.array(lossO), np.array(numO), parameterS / parameterO


#saveJacobian(n_el=20, n_per_el=3)
#testAlgorithm('forward.h5', n=5)


"""
measurements_0 = cp.random.randint(0, 20, (1500, 4))

while cp.sum(cp.equal(measurements_0[:, :, None], measurements_0[:, None, :])) > 4 * len(measurements_0):
	measurements_0 = cp.random.randint(0, 20, (35, 4))
v_meas = 10 * np.ones(len(measurements_0)) * np.random.randn()

#print("\n --- Saving Jacobian --- \n")
#saveJacobian()
#print("\n --- Finding next pair --- \n")
#findNextPair('forward.h5', 20, el_dist=1)

#saveJacobian()
#print("\n --- Optimising the batch --- \n")
#batchOptimize('forward.h5', batchSize=90)
#print("\n --- Testing the algorithm Uwu --- \n")
#testAlgorithm('forward.h5', n=5)


findNextPair('forward__.h5', 20, el_dist=1)
saveJacobian()
#print("\n --- Optimising the batch --- \n")
batchOptimize('forward_.h5', batchSize=90)
#print("\n --- Testing the algorithm Uwu --- \n")
testAlgorithm('forward_.h5', n=5)

iterations = 5
lossS, numS = [], []
lossO, numO = [], []
for i in range(iterations):
	lossSugg, numSugg, lossOrd, numOrd, _ = testAlgorithm('forward__.h5', n=10, pert=0.3141784084620273, cutoff=0.6357154598817543, p_influence=-8.74675301016891, p_rec=0.5546683314342027)
	minim = np.amin(lossSugg[:12])
	print('minimal of', i, '=', minim)
	lossOrd = lossOrd[numOrd <= np.amax(numSugg)]
	numOrd = numOrd[numOrd <= np.amax(numSugg)]

	lossS.append(lossSugg[:12]/minim)
	numS.append(numSugg[:12])
	lossO.append(lossOrd[:12]/minim)
	numO.append(numOrd[:12])
'''
'''
try:
	lossO = np.array(lossO)
	lossS = np.array(lossS)
	lossO = np.sum(lossO, axis=0)
	lossS = np.sum(lossS, axis=0)
	print('lossO' , lossO)
	print('lossS' , lossS)
except:
	lossO0 = np.sum(lossO[:, 0])
	print('lossO0 = ', lossO0)
	lossS0 = np.sum(lossS[:, 0])
	print('lossS0 = ', lossS0)
	lossO1 = np.sum(lossO[:, 1])
	print('lossO1 = ', lossO1)
	lossS1 = np.sum(lossS[:, 1])
	print('lossS1 = ', lossS1)
	lossO2 = np.sum(lossO[:, 2])	
	print('lossO2 = ', lossO2)
	lossS2 = np.sum(lossS[:, 2])
	print('lossS2 = ', lossS2)
	lossO3 = np.sum(lossO[:, 3])
	print('lossO3 = ', lossO3)
	lossS3 = np.sum(lossS[:, 3])
	print('lossS3 = ', lossS3)
	lossO4 = np.sum(lossO[:, 4])
	print('lossO4 = ', lossO4)
	lossS4 = np.sum(lossS[:, 4])
	print('lossS4 = ', lossS4)
	lossO5 = np.sum(lossO[:, 5])
	print('lossO5 = ', lossO5)
	lossS5 = np.sum(lossS[:, 5])
	print('lossS5 = ', lossS5)
	lossO6 = np.sum(lossO[:, 6])
	print('lossO6 = ', lossO6)
	lossS6 = np.sum(lossS[:, 6])
	print('lossS6 = ', lossS6)
	lossO7 = np.sum(lossO[:, 7])
	print('lossO7 = ', lossO7)
	lossS7 = np.sum(lossS[:, 7])
	print('lossS7 = ', lossS7)
	lossO8 = np.sum(lossO[:, 8])
	print('lossO8 = ', lossO8)
	lossS8 = np.sum(lossS[:, 8])
	print('lossS8 = ', lossS8)
	lossO9 = np.sum(lossO[:, 9])
	print('lossO9 = ', lossO9)
	lossS9 = np.sum(lossS[:, 9])
	print('lossS9 = ', lossS9)
	lossO10 = np.sum(lossO[:, 10])	
	print('lossO10 = ', lossO10)
	lossS10 = np.sum(lossS[:, 10])
	print('lossS10 = ', lossS10)
	lossO11 = np.sum(lossO[:, 11])
	print('lossO11 = ', lossO11)
	lossS11 = np.sum(lossS[:, 11])
	print('lossS11 = ', lossS11)

'''
'''
fig1, ax1 = plt.subplots()
ax1.plot(numS[0], lossS[0], 'r.')
ax1.plot(numS[1], lossS[1], 'b.')
ax1.plot(numS[2], lossS[2], 'g.')
ax1.plot(numS[3], lossS[3], 'm.')
ax1.plot(numS[4], lossS[4], 'c.')

#ax1.set_title('L2 Loss of Our Algorithm Selected Measurement Technique')
ax1.set_title('L2 Loss of Electrode Selection Algorithm')
ax1.set_xlabel('Number of Measurements Made')
ax1.set_ylabel('(L2 Loss)$^2$')

fig2, ax2 = plt.subplots()
ax2.plot(numO[0], lossO[0], 'r.')
ax2.plot(numO[1], lossO[1], 'b.')
ax2.plot(numO[2], lossO[2], 'g.')
ax2.plot(numO[3], lossO[3], 'm.')
ax2.plot(numO[4], lossO[4], 'c.')
#ax2.set_title('L2 Loss of a Common Measurement Technique: El. Dist. = 10, Step = 1')
#ax2.set_xlabel('Number of Measurements')
ax2.set_title('L2 Loss of Common Measurement Pattern (opposite)')
ax2.set_xlabel('Number of Measurements Made')
ax1.set_ylabel('(L2 Loss)$^2$')
plt.show()
'''

'''
perm = solveGREIT()

fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(np.real(perm), interpolation='none', cmap=plt.cm.viridis, origin='lower', extent=[-1,1,-1,1])
fig.colorbar(im)
ax.axis('equal')

plt.show()
"""