'''
6 April 2020

Python file that contains the custom algorithm that finds the most optimal source/sink pair

and the most useful voltage measurement electrode pairs.

by Ivo Mihov and Vasil Avramov 

in collaboration with Artem Mischenko and Sergey Slizovskiy 

from Solid State Physics Group 

at the University of Manchester

'''

import numpy as np
import meshing
import cupy as cp
from pyeit.eit.fem import Forward as FEM_normal
from pyeit.eit.fem_forallmeas import Forward
from pyeit.eit.fem_for_given_meas import Forward as FEM
import h5py as h5
import greit_rec_training_set as train

def volt_mat_all(ne):
	#generates all possible pairs of voltage measurements
	step_arr = cp.arange(ne)
	parser = None
	A = cp.arange(ne)
	M = cp.dot(cp.ones(ne)[:,None], A[None, :]) % ne
	N = (M + step_arr[:, None]) % ne	
	pair_mat = cp.stack((N, M), axis=-1)
	pair_mat = pair_mat.reshape(((ne)**2, 2))

	ind = pair_mat[:,0] < pair_mat[:, 1]

	return pair_mat[ind].astype(int)

def get_total_map(rec, voltages, h_mat, npix=64, pert=0.5, p_influence=-10., p_rec=10.):
	v_pert = np.empty(shape=(len(voltages), len(voltages)))
	perturbing_mat = np.ones((len(voltages), len(voltages))) + pert * np.identity(len(voltages))
	v_pert[:] = np.dot(perturbing_mat, np.diag(voltages))
	influence_mat = -np.dot(h_mat, v_pert).reshape(npix, npix, len(voltages)) - rec[:, :, None]
	influence_mat = np.absolute(influence_mat)
	influence_mat = np.sum(influence_mat, axis=2)
	grad_mat =  np.linalg.norm(np.gradient(rec), axis = 0)

	if np.amin(rec) <= -1.:
		rec = - rec / np.amin(rec)
	
	rec = np.log(rec + 1.0000001)
	#rec[(rec < 1.e-3) * (rec > 0.)] = 1.e-3
	#rec[(rec > -1.e-3) * (rec < 0.)] = -1.e-3
	#p_influence /= 1e3
	return grad_mat * influence_mat ** p_influence * rec ** p_rec, grad_mat, rec

def get_centre_squares(npix=64, a = 2., centre=None):

	if centre is None:
		centre = [0, 0]
	C_sq = np.empty((npix, npix, 2), dtype='f4')
	# initialising the j vector to prepare for tiling
	j = np.arange(npix)
	# tiling j to itself npix times (makes array shape (npix, npix))
	j = np.tile(j, (npix, 1))
	# i and j are transposes of each other	
	i = j
	j = j.T
	# assigning values to C_sq 
	C_sq[i, j, :] = np.transpose([a / 2 * ((2 * i + 1) / npix - 1) + centre[0], a / 2 * ((2 * j + 1) / npix - 1) + centre[1]])

	return C_sq

def find_all_distances(rec, h_mat, el_coords, n_el, a = 2., npix = 64, cutoff = 0.8):
	
	c_sq = get_centre_squares(npix = 64, a = 2.)
	#print(el_coords)
	if el_coords.shape[0] > n_el:
		el_coords = np.mean(el_coords.reshape((n_el, int(el_coords.shape[0] / n_el), 2)), axis=1)
	#print(el_coords)
	all_meas_pairs = cp.asnumpy(volt_mat_all(n_el))
	#print(all_meas_pairs.shape)
	const_X_index = ( (el_coords[all_meas_pairs[:, 0], 0] - el_coords[all_meas_pairs[:, 1], 0]) == 0 )
	const_Y_arr = np.sort(el_coords[all_meas_pairs[const_X_index], 1], axis=1)

	all_slopes = np.empty(all_meas_pairs.shape[0])
	all_const =  np.empty(all_meas_pairs.shape[0])
	all_slopes[~const_X_index] = (el_coords[all_meas_pairs[~const_X_index, 0], 1] - el_coords[all_meas_pairs[~const_X_index, 1], 1]) / (
					el_coords[all_meas_pairs[~const_X_index, 0], 0] - el_coords[all_meas_pairs[~const_X_index, 1], 0])
	all_const[~const_X_index] = el_coords[all_meas_pairs[~const_X_index, 1], 1] - el_coords[all_meas_pairs[~const_X_index, 1], 0] * all_slopes[~const_X_index]
	
	p_st = np.empty(shape=(all_meas_pairs.shape[0],2))
	p_st[:, 0] = np.amin(el_coords[all_meas_pairs,0], axis = 1)
	p_st[:, 1] = all_slopes * p_st[:, 0] + all_const

	p_st[const_X_index, 0] = el_coords[all_meas_pairs[const_X_index][:, 0], 0]
	p_st[const_X_index, 1] = const_Y_arr[:, 0]

	p_end = np.empty(shape=(all_meas_pairs.shape[0],2))
	p_end[:, 0] = np.amax(el_coords[all_meas_pairs,0], axis = 1)
	p_end[:, 1] = all_slopes * p_end[:, 0] + all_const

	p_end[const_X_index, 0] = el_coords[all_meas_pairs[const_X_index][:, 0], 0]
	p_end[const_X_index, 1] = const_Y_arr[:, 1]

	indices = (np.abs(np.cross((p_end - p_st)[:, None, None, :],
								   np.transpose(np.array([p_st[:, 0][:, None, None] - c_sq[None, :, :, 0],
								   			 p_st[:, 1][:, None, None] - c_sq[None, :, :, 1]]), axes=(1, 2, 3, 0)))
					   / np.linalg.norm(p_end - p_st, axis=1)[:, None, None])
					   < a / (npix * np.sqrt(2)))

	return indices, all_meas_pairs


def findNextPair(fileJac, ne, anomaly=None, el_dist=None, meas=None, a=2., npix=64, pert=0.5, p_influence=-10., p_rec=10.):	
	# find indices in the already calculated const. permitivity Jacobian (CPJ)
	if meas is None:
		if el_dist is None:
			el_dist = np.random.randint(1, 20)
		A = cp.arange(ne)
		B = (cp.arange(ne) + el_dist) % ne
		ex_mat = cp.stack((A, B), axis=1)
		#test to see how different excitation matrices will affect the sensitivity in different areas
		#even_ind = (np.arange((len(A))) % 2) == 0
		#ex_mat = ex_mat[even_ind]

		volt_mat_all, ex_mat, ind_new = voltMeterwStep(ne, ex_mat, step_arr=10)
		measurements = cp.concatenate((ex_mat, volt_mat_all), axis=1)
	else:
		measurements = meas
	#print(measurements)
	rec, h_mat, voltages, volt_const, tr, num = simulateMeasurements(fileJac, anomaly=anomaly, measurements=measurements)
	#volts = volts.reshape((len(volts), 1))

	v_pert = np.empty(shape=(len(voltages), len(voltages)))
	perturbing_mat = np.ones((len(voltages), len(voltages))) + pert * np.identity(len(voltages))
	v_pert[:] = np.dot(perturbing_mat, np.diag(voltages))
	influence_mat = -np.dot(h_mat, v_pert).reshape(npix, npix, len(voltages)) - rec[:, :, None]
	influence_mat = np.absolute(influence_mat)
	influence_mat = np.sum(influence_mat, axis=2)

	_, el_coords = meshing.fix_electrodes_multiple(centre=None, edgeX=0.1, edgeY=0.1, a=2, b=2, ppl=ne, el_width=0.2, num_per_el=3)
	pixel_indices, voltage_all_possible = find_all_distances(rec, h_mat, el_coords, 20, cutoff = 0.8)
	total_map, grad_mat, rec_log = np.abs(get_total_map(rec, voltages, h_mat, pert=pert, p_influence=p_influence, p_rec=p_rec))
	'''
	plt.figure(3)
	im3 = plt.imshow(grad_mat.astype(float), cmap=plt.cm.viridis, origin='lower', extent=[-1, 1, -1, 1])
	plt.title("Gradient of reconstruction")
	plt.colorbar(im3)
	plt.figure(4)
	im4 = plt.imshow(rec_log.astype(float), cmap=plt.cm.viridis, origin='lower', extent=[-1, 1, -1, 1])
	plt.title("Reconstruction logarithmic")
	plt.colorbar(im4)
	plt.figure(5)
	im5 = plt.imshow(influence_mat.astype(float), cmap=plt.cm.viridis, origin='lower', extent=[-1, 1, -1, 1])
	plt.title("Influence Map")
	plt.colorbar(im5)
	plt.figure(6)
	im6 = plt.imshow(total_map.astype(float), cmap=plt.cm.viridis, origin='lower', extent=[-1, 1, -1, 1])
	plt.title("Total Map")
	plt.colorbar(im6)'''
	
	total_maps_along_lines = total_map[None] * pixel_indices
	proximity_to_boundary = np.sum(total_maps_along_lines, axis=(1, 2)) / np.sum(pixel_indices, axis=(1, 2))
	proposed_ex_line = voltage_all_possible[np.argsort(proximity_to_boundary)[::-1]][:10]
	#print(proposed_ex_line)

	#volt_measurements = findNextVoltagePair(proposed_ex_line, fileJac, total_map, npix, 0.85)

	'''
	# optional: weight the sensitivity by its distance from the center to avoid focusing only on edge areas
	xm = np.arange(64)
	xm, ym= np.meshgrid(xm, xm)
	print(xm, ym)
	xm = (xm * a / npix + a / (2 * npix)) - a / 2
	ym = (ym * a / npix + a / (2 * npix)) - a / 2
	weight = xm**2 + ym**2
	print(weight[30:34, 30:34])
	influence_mat *= weight
	
	# old measurement selection
	zones = np.less(influence_mat, 2 * np.amin(influence_mat))
	#zones = np.greater(influence_mat, np.amax(influence_mat)/1.2)
	zone_index = np.where(zones)

	zone_index = np.concatenate((zone_index[1][:, None], zone_index[0][:, None]), axis=1)
	zone_coords = (zone_index * a / npix + a / (2 * npix)) - a / 2
	avg_coords = np.mean(zone_coords, axis=0)
	closest = np.argsort(np.linalg.norm(zone_coords - avg_coords[None], axis=1))
	closest_coords = zone_coords[closest[:3]]

	
	# for one coordinate
	closest_coords = np.array(np.where(influence_mat == np.amin(influence_mat)))
	closest_coords = (closest_coords * a / npix + a / (2 * npix)) - a / 2
	chosen_pair = findMinimumTriArea(closest_coords)
	'''
	#plt.show()
	'''
	plt.plot(closest_coords[:, 0], closest_coords[:, 1], 'rx')
	plt.plot(avg_coords[0], avg_coords[1], 'gx')
	plt.show()'''	
	return proposed_ex_line, rec, tr, num, total_map

def findNextVoltagePair(new_ex_line, jac_file, total_map, number_of_voltages, counter, npix=64, cutoff=0.97):
	
	file = h5.File(jac_file, 'r')
	p = file['p'][()]
	t = file['t'][()]
	jac = file['jac'][()]
	meas = file['meas'][()]
	file.close()
	#region of interest index
	roi_index = total_map > cutoff * np.amax(total_map)
	C_sq = get_centre_squares()
	square_to_tri_index = np.empty((npix, npix,  t.shape[0]), dtype=bool)
	C_tri = np.mean(p[t], axis=1)
	distance_all = np.linalg.norm(C_sq[:, :, None] - C_tri[None, None, :], axis=3)
	weight = 1./(1 + np.exp(1.*(distance_all - 0.)))
	weight[~roi_index] = 0.
	weight_on_each_tri = np.sum(weight, (0, 1))
	#new_ex_line = np.sort(new_ex_line)
	index_source = new_ex_line[0] == meas[:, 0]
	index_sink = new_ex_line[1] == meas[:, 1]
	index_current = index_source * index_sink
	#assert np.sum(index_current) == (n_el - 2) * (n_el - 3) / 2
	jac_of_interest = jac[index_current]
	jac_weighted = jac_of_interest * weight_on_each_tri[None, :]
	voltage_pair_initial_index = np.argsort(np.sum(jac_weighted, axis=1))[::-1]
	#print(meas[index_current][voltage_pair_initial_index])
	voltage_measurement_predictions = meas[index_current, 2:][voltage_pair_initial_index]
	#print(voltage_measurement_predictions)
	return meas[index_current, 2:][voltage_pair_initial_index][int(counter * number_of_voltages) : int((counter + 1) * number_of_voltages)]


def circleMask(npix, a, pc=[0, 0]):
	C_sq = np.empty((npix, npix, 2), dtype='f4')
	j = np.arange(npix)
	j = np.tile(j, (npix, 1))
	i = j
	j = j.T
	C_sq[i, j, :] = np.transpose([a / 2 * ((2 * i + 1) / npix - 1) + pc[0], a / 2 * ((2 * j + 1) / npix - 1) + pc[1]])
	mask = np.power(C_sq[:, :, 0], 2) + np.power(C_sq[:, :, 1], 2) < np.power(a/2, 2)

	return mask
	
def findMinimumTriArea(closest_coords):
	el_coords = train.fix_electrodes(edgeX=0.2, edgeY=0.2, a=2, b=2, ppl=20)[1]
	ex_mat_all = train.generateExMat(ne=20)
	coord_mat = el_coords[ex_mat_all]

	#tri_area_all = np.empty(len(ex_mat_all))
	'''
	# for one coordinate
	tri_area = 0.5 * np.absolute(coord_mat[:, 0, 0] * (coord_mat[:, 1, 1] - closest_coords[0]) - coord_mat[:, 0, 1] * (coord_mat[:, 1, 0] - closest_coords[1]) + coord_mat[:, 1, 0]*closest_coords[0] - coord_mat[:, 1, 1] * closest_coords[1] )
	min_area_ind  = np.argsort(tri_area)
	'''
	tri_area_sum = np.sum(0.5 * np.absolute(coord_mat[:, None, 0, 0] * (coord_mat[:, None, 1, 1] - closest_coords[None, :, 1]) - coord_mat[:, None, 0, 1] * (coord_mat[:, None, 1, 0] - closest_coords[None, :, 0]) + coord_mat[:, None, 1, 0]*closest_coords[None, :, 1] - coord_mat[:, None, 1, 1] * closest_coords[None, :, 0] ), axis=1)
	#print(np.array_equal(tri_area_all[:, 1], tri_area))
	min_area_ind = np.argsort(tri_area_sum)
	
	ex_mat_pred = ex_mat_all[min_area_ind[:10]]
	#print('Next src/sink recommendation (from 1 to n_el starting from upper left corner clockwise):', ex_mat_pred + 1)
	return ex_mat_pred

def voltMeterwStep(ne, ex_mat, step_arr=None, parser=None):
    '''
    
    Returns all measurements with this step_arr and ex_mat

    takes:

    ex_mat - array shape (n_source/sinks, 2) - excitation matrix with source and sink for each measurement
    step_arr - array shape (n_source/sinks) - step between measuring electrodes for each source/sink pair
    parser - string

    returns:

    pair_mat - array shape (n_measurements, 2) - matrix with all possible meas. electrode combinations
    ind_new - array shape (n_measurements) - helper array

    '''        
    if step_arr is None:
        step_arr = 1 + cp.arange((ex_mat.shape[0])) % (ne)
    elif type(step_arr) is int:
    	step_arr = step_arr * cp.ones(ex_mat.shape[0]) % ne

    drv_a = ex_mat[:, 0]
    drv_b = ex_mat[:, 1]
    i0 = drv_a if parser == 'fmmu' else 0
    A = cp.arange(ne)
    
    #M = cp.dot(cp.ones(ex_mat.shape[0])[:,None], A[None, :]) % self.ne
    #N = (M + step_arr[:, None]) % self.ne

    M = cp.arange(ex_mat.shape[0] * ne) % ne
    N = (M.reshape((ex_mat.shape[0], ne)) + step_arr[:, None]) % ne
    pair_mat = cp.stack((N.ravel(), M), axis=-1)

    #ind_new = cp.arange(pair_mat.shape[0]) % ex_mat.shape[0]
    ind_new = cp.arange(ex_mat.shape[0])        
    ind_new = cp.tile(ind_new, (ne, 1)).T.ravel()
    #print('before indtest', ind_new[20:70])
    nz2 = cp.where(pair_mat == drv_a[ind_new, None])
    nz3 = cp.where(pair_mat == drv_b[ind_new, None])
    #print(ind_new)
    ind_ = cp.arange(pair_mat.shape[0])
    ind_fin = cp.sum(ind_[:, None] == nz2[0][None], axis=1)
    ind_fin2 = cp.sum(ind_[:, None] == nz3[0][None], axis=1)

    ind_test = cp.less((ind_fin + ind_fin2), 0.5 * cp.ones(len(ind_fin)))

    pair_mat = pair_mat[ind_test, :]
    ind_new = ind_new[ind_test]
    sort_index = cp.argsort(ind_new)

    #print('after indtest', ind_new[20:70])
    #meas = cp.concatenate((ex_mat[ind_new], pair_mat), axis=1)
    #print(meas[20:70])
    return pair_mat, ex_mat[ind_new], ind_new


def simulateMeasurements(fileJac, anomaly=0, measurements=None, v_meas=None, n_el=20, n_per_el=3, n_pix=64, a=2.):
	# extract const permittivity jacobian and voltage (& other)
	file = h5.File(fileJac, 'r')

	meas = file['meas'][()]
	new_ind = file['new_ind'][()]
	p = file['p'][()]
	t = file['t'][()]
	file.close()
	# initialise const permitivity and el_pos variables
	perm = np.ones(t.shape[0], dtype=np.float32)
	el_pos = np.arange(n_el * n_per_el).astype(np.int16)
	mesh_obj = {'element': t,
				'node':	p,
				'perm':	perm}

	#for testing
	if measurements is None:
		el_dist = np.random.randint(1, 20)
		ex_mat = (cp.concatenate((cp.arange(20)[None], (cp.arange(20) + el_dist)[None])) % 20).T
		#print(ex_mat.shape)
		fem_all = Forward(mesh_obj, el_pos)
		measurements = fem_all.voltMeter(ex_mat)
		#ex_mat = mesurements[1]
		measurements = cp.concatenate((measurements[1], measurements[0]), axis=1)
		#print(measurements.shape)
	# list all possible active/measuring electrode permutations of this measurement
	meas = cp.array(meas)
	# find their indices in the already calculated const. permitivity Jacobian (CPJ)
	measurements = cp.array(measurements)
	measurements_0 = cp.amin(measurements[:, :2], axis=1)
	measurements_1 = cp.amax(measurements[:, :2], axis=1)
	measurements_2 = cp.amin(measurements[:, 2:], axis=1)
	measurements_3 = cp.amax(measurements[:, 2:], axis=1)
	measurements = cp.empty((len(measurements), 4))
	measurements[:, 0] = measurements_0
	measurements[:, 1] = measurements_1
	measurements[:, 2] = measurements_2
	measurements[:, 3] = measurements_3
	index = (cp.sum(cp.equal(measurements[:, None, :], meas[None, :, :]), axis=2) == 4)
	index = cp.where(index)
	ind = cp.unique(index[1])
	i = cp.asnumpy(ind)
	j = index[0]
	mask = np.zeros(len(meas), dtype=int)
	mask[i] = 1
	mask = mask.astype(bool)
	# take a slice of Jacobian, voltage readings and B matrix
	file = h5.File(fileJac, 'r')
	jac = file['jac'][mask, :][()]
	v = file['v'][mask][()]
	b = file['b'][mask, :][()]
	file.close()
	pde_result = train.namedtuple("pde_result", ['jac', 'v', 'b_matrix'])
	f = pde_result(jac=jac,
				   v=v,
				   b_matrix=b)
	
	# simulate voltage readings if not given
	if v_meas is None:
		if np.isscalar(anomaly):
			print("generating new anomaly")
			anomaly = train.generate_anoms(a, a)
		true = train.generate_examplary_output(a, int(n_pix), anomaly)
		mesh_new = train.set_perm(mesh_obj, anomaly=anomaly, background=1)
		fem = FEM(mesh_obj, el_pos, n_el)
		new_ind = cp.array(new_ind)
		f2, raw = fem.solve_eit(volt_mat_all=meas[ind, 2:], new_ind=new_ind[ind], ex_mat=meas[ind, :2], parser=None, perm=mesh_new['perm'].astype('f8'))
		v_meas = f2.v
		'''
		#plot
		fig = plt.figure(3)
		x, y = p[:, 0], p[:, 1]
		ax1 = fig.add_subplot(111)
		# draw equi-potential lines
		print(raw.shape)
		raw = cp.asnumpy(raw[5]).ravel()
		vf = np.linspace(min(raw), max(raw), 32)
		ax1.tricontour(x, y, t, raw, vf, cmap=plt.cm.viridis)
		# draw mesh structure
		ax1.tripcolor(x, y, t, np.real(perm),
					  edgecolors='k', shading='flat', alpha=0.5,
					  cmap=plt.cm.Greys)

		ax1.plot(x[el_pos], y[el_pos], 'ro')
		for i, e in enumerate(el_pos):
			ax1.text(x[e], y[e], str(i+1), size=12)
		ax1.set_title('Equipotential Lines of Uniform Permittivity')
		# clean up
		ax1.set_aspect('equal')
		ax1.set_ylim([-1.2, 1.2])
		ax1.set_xlim([-1.2, 1.2])
		fig.set_size_inches(6, 6)
		#plt.show()'''
	elif len(measurements) == len(v_meas):
		measurements = np.array(measurements)
		v_meas = np.array(v_meas[j[:len(ind)]])
	else:
		raise ValueError('Sizes of arrays do not match (have to have voltage reading for each measurement). If you don\'t have readings, leave empty for simulation.')
	print('Number of measurements:', len(v_meas), len(f.v))

	# now we can use the real voltage readings and the GREIT algorithm to reconstruct
	greit = train.greit.GREIT(mesh_obj, el_pos, f=f, ex_mat=(meas[index[1], :2]), step=None)
	greit.setup(p=0.2, lamb=0.01, n=n_pix)
	h_mat = greit.H
	reconstruction = greit.solve(v_meas, f.v).reshape(n_pix, n_pix)
	
	# optional: see reconstruction
	'''
	plt.figure(1)
	im1 = plt.imshow(reconstruction, cmap=plt.cm.viridis, origin='lower', extent=[-1, 1, -1, 1])
	plt.title("Reconstruction")
	plt.colorbar(im1)
	plt.figure(2)
	im2 = plt.imshow(true, cmap=plt.cm.viridis, origin='lower', extent=[-1, 1, -1, 1])
	plt.colorbar(im2)
	plt.title("True Image")
	plt.show()
	'''
	return reconstruction, h_mat, v_meas, f.v, true, len(v_meas)