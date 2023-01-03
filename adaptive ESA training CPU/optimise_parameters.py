'''
6 April 2020

Python file that optimises the parameters of the custom measurement optimisation algorithm

and the most useful voltage measurement electrode pairs.

by Ivo Mihov and Vasil Avramov 

in collaboration with Artem Mischenko and Sergey Slizovskiy 

from Solid State Physics Group at the University of Manchester


'''

import numpy as np
import cupy as cp
import csv
from datetime import datetime
import meshing
from controls import *
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from pyeit.mesh import set_perm
from skopt import gp_minimize, dump, load
import matplotlib.pyplot as plt
from measurement_optimizer import volt_mat_all


cp.cuda.Device(0).use()

def L2_Loss(rec, im):
    return np.sum((rec - im) ** 2)

class OptimiseMeasurementAlgorithm(object):
    def __init__(self, n_samples=2e4, ne=20, a=2., npix=64, n_per_el=2):
        init_t = time()
        self.ne = ne
        self.a = a
        self.npix = npix
        self.n_per_el = n_per_el
        self.C_sq = get_centre_squares(npix = self.npix)
        self.all_meas = self.getJacobian()
        self.C_tri = np.mean(self.all_meas.mesh_obj['node'][self.all_meas.mesh_obj['element']], axis=1)
        self.distance_all = np.linalg.norm(self.C_sq[:, :, None] - self.C_tri[None, None, :], axis=3)
        self.weight = 1./(1 + np.exp(1.*(self.distance_all - 0.)))

        self.fwd = self.all_meas.forward_obj
        self.w_mat = self.getWeightSigmoid()
        self.generate_training_set(n_samples)
        _, self.el_coords = meshing.fix_electrodes_multiple(centre=None, edgeX=0.08, edgeY=0.08, a=self.a, b=self.a, ppl=self.ne, el_width=0.04, num_per_el=self.n_per_el)
        self.pixel_indices, self.voltage_all_possible = self.find_all_distances()
        self.sum_of_pixels_indices = np.sum(self.pixel_indices, axis=(1, 2))
        print('Time for init:', time() - init_t)

    def getWeightSigmoid(self, ext_ratio=0, gc=False, s=20., ratio=0.1):
        x, y = self.all_meas.mesh_obj['node'][:, 0], self.all_meas.mesh_obj['node'][:, 1]
        x_min, x_max = min(x), max(x)
        y_min, y_max = min(y), max(y)
        x_ext = (x_max - x_min) * ext_ratio
        y_ext = (y_max - y_min) * ext_ratio
        xv, xv_step = np.linspace(x_min-x_ext, x_max+x_ext, num=self.npix,
                                  endpoint=False, retstep=True)
        yv, yv_step = np.linspace(y_min-y_ext, y_max+y_ext, num=self.npix,
                                  endpoint=False, retstep=True)
        # if need grid correction
        if gc:
            xv = xv + xv_step / 2.0
            yv = yv + yv_step / 2.0
        xg, yg = np.meshgrid(xv, yv, sparse=False, indexing='xy')
        # pts_edges = pts[el_pos]
        cv = ConvexHull(self.all_meas.mesh_obj['node'])
        hull_nodes = cv.vertices
        pts_edges = self.all_meas.mesh_obj['node'][hull_nodes, :]


        # 1. create mask based on meshes
        points = np.vstack((xg.flatten(), yg.flatten())).T

        # 2. extract edge points using el_pos
        path = Path(pts_edges, closed=False)
        mask = ~(path.contains_points(points))
        xy = np.mean(self.all_meas.mesh_obj['node'][self.all_meas.mesh_obj['element']], axis=1)
        xyi = np.vstack((xg.flatten(), yg.flatten())).T

        d0 = xy[:, 0].ravel()[:, np.newaxis] - xyi[:, 0].ravel()[np.newaxis, :]
        d1 = xy[:, 1].ravel()[:, np.newaxis] - xyi[:, 1].ravel()[np.newaxis, :]
        distance = np.hypot(d0, d1)
        d_max = np.max(distance)
        distance = 5.0 * distance / d_max
        r0 = 5.0 * ratio
        weight = 1./(1 + np.exp(s*(distance - r0)))
        # normalized
        w_mat = weight / np.sum(weight, axis=0)
        return w_mat

    def getJacobian(self):
          # number electrodes
          el_pos = np.arange(self.ne * self.n_per_el)
          # create an object with the meshing characteristics to initialise a Forward object
          #mesh_obj = train.mesh(self.ne)
          #mesh_params = [0.045, 2000, 3.67, 3.13]
          #mesh_params = [0.054, 2000, 9.34, 1.89]
          mesh_params = [0.08, 2000, 0.45, 0.3]
          #mesh_params = [0.068,2000,6.98,3.25]
          #mesh_params = [0.081,2500,9.55,3.14]
          #mesh_obj = train.mesh(n_el=self.ne, num_per_el=self.n_per_el, edge= 0.08, el_width=0.04, mesh_params=mesh_params)
          mesh_obj = train.mesh(n_el=self.ne, num_per_el=self.n_per_el, edge= 0.2, el_width=0.1, mesh_params=mesh_params)
          fwd = Forward(mesh_obj, el_pos, self.ne)
          self.ex_mat_all = train.generateExMat(ne=self.ne)
          f, meas, new_ind = fwd.solve_eit(ex_mat=self.ex_mat_all, perm=fwd.tri_perm)
          #print(f)
          #ind = np.arange(len(meas))
          ind = new_ind
          #pde_result = train.namedtuple("pde_result", ['jac', 'v', 'b_matrix'])
          #ind = np.lexsort((meas[:,3], meas[:, 2], meas[:, 1], meas[:, 0]))
  
          # f = pde_result(jac=f.jac[ind],
          #                v=f.v[ind],
          #                b_matrix=f.b_matrix) # b_matrix=f.b_matrix[ind]
          all_meas_object = train.namedtuple("all_meas_object", ['f', 'meas', 'new_ind', 'mesh_obj','forward_obj'])
          all_meas = all_meas_object(f=f,
                         meas=meas,
                         new_ind=ind,
                         mesh_obj=mesh_obj,
                         forward_obj=fwd)
          # all_meas = all_meas_object(f=f,
          #                            meas=meas[ind],
          #                            new_ind=new_ind[ind],
          #                            mesh_obj=mesh_obj,
          #                            forward_obj=fwd)
          return all_meas

    def h_mat(self, mask, p=0.2, lamb=0.01):
        jac = self.all_meas.f.jac[mask, :]
        j_j_w = np.dot(jac, jac.T)
        #print('There are nans in jac: ', np.isnan(jac).any())
        r_mat = np.diag(np.power(np.diag(j_j_w), p))
        jac_inv = np.linalg.inv(j_j_w + lamb*r_mat)
        h_mat = np.einsum('jk, lj, lm', self.w_mat, jac, jac_inv, optimize='optimal')
        return h_mat

    def get_all_voltages(self, anomaly):
        mesh_new = set_perm(self.all_meas.mesh_obj, anomaly=anomaly, background=None)
        voltage_readings, measurements, new_ind = self.fwd.solveAnomaly(ex_mat=self.ex_mat_all, step=1, perm=mesh_new['perm'], parser=None)
        ind = np.lexsort((measurements[:,3], measurements[:, 2], measurements[:, 1], measurements[:, 0]))
        return voltage_readings[ind], measurements[ind]

    def extract_voltages(self, all_measurements, measurements):
        meas_test_src = np.sort(measurements[:, :2], axis=1)
        meas_test_volt = np.sort(measurements[:, 2:], axis=1)
        meas_test = np.concatenate((meas_test_src, meas_test_volt), axis=1)
        '''
        measurements_0 = np.amin(measurements[:, :2], axis=1)
        measurements_1 = np.amax(measurements[:, :2], axis=1)
        measurements_2 = np.amin(measurements[:, 2:], axis=1)
        measurements_3 = np.amax(measurements[:, 2:], axis=1)
        measurements = np.empty((len(measurements), 4))
        measurements[:, 0] = measurements_0
        measurements[:, 1] = measurements_1
        measurements[:, 2] = measurements_2
        measurements[:, 3] = measurements_3

        print(np.array_equiv(meas_test, measurements))
        '''

        index = np.equal(measurements[:, None, :], all_measurements[None, :, :]).all(2)
        index = np.where(index)
        ind = np.unique(index[1])
        mask = np.zeros(len(all_measurements), dtype=int)
        mask[ind] = 1
        mask = mask.astype(bool)

        return mask

    def find_all_distances(self):
        
        #print(el_coords)
        if self.el_coords.shape[0] > self.ne:
            self.el_coords = np.mean(self.el_coords.reshape((self.ne, int(self.el_coords.shape[0] / self.ne), 2)), axis=1)
        #print(el_coords)
        all_meas_pairs = cp.asnumpy(volt_mat_all(self.ne))
        #print(all_meas_pairs.shape)
        const_X_index = ( (self.el_coords[all_meas_pairs[:, 0], 0] - self.el_coords[all_meas_pairs[:, 1], 0]) == 0 )
        const_Y_arr = np.sort(self.el_coords[all_meas_pairs[const_X_index], 1], axis=1)

        all_slopes = np.empty(all_meas_pairs.shape[0])
        all_const =  np.empty(all_meas_pairs.shape[0])
        all_slopes[~const_X_index] = (self.el_coords[all_meas_pairs[~const_X_index, 0], 1] - self.el_coords[all_meas_pairs[~const_X_index, 1], 1]) / (
                        self.el_coords[all_meas_pairs[~const_X_index, 0], 0] - self.el_coords[all_meas_pairs[~const_X_index, 1], 0])
        all_const[~const_X_index] = self.el_coords[all_meas_pairs[~const_X_index, 1], 1] - self.el_coords[all_meas_pairs[~const_X_index, 1], 0] * all_slopes[~const_X_index]
        
        p_st = np.empty(shape=(all_meas_pairs.shape[0],2))
        p_st[:, 0] = np.amin(self.el_coords[all_meas_pairs,0], axis = 1)
        p_st[:, 1] = all_slopes * p_st[:, 0] + all_const

        p_st[const_X_index, 0] = self.el_coords[all_meas_pairs[const_X_index][:, 0], 0]
        p_st[const_X_index, 1] = const_Y_arr[:, 0]

        p_end = np.empty(shape=(all_meas_pairs.shape[0],2))
        p_end[:, 0] = np.amax(self.el_coords[all_meas_pairs,0], axis = 1)
        p_end[:, 1] = all_slopes * p_end[:, 0] + all_const

        p_end[const_X_index, 0] = self.el_coords[all_meas_pairs[const_X_index][:, 0], 0]
        p_end[const_X_index, 1] = const_Y_arr[:, 1]

        indices = (np.abs(np.cross((p_end - p_st)[:, None, None, :],
                                       np.transpose(np.array([p_st[:, 0][:, None, None] - self.C_sq[None, :, :, 0],
                                                    p_st[:, 1][:, None, None] - self.C_sq[None, :, :, 1]]), axes=(1, 2, 3, 0)))
                           / np.linalg.norm(p_end - p_st, axis=1)[:, None, None])
                           < self.a / (self.npix * np.sqrt(2)))

        return indices, all_meas_pairs

    def get_total_map(self, rec, voltages, h_mat, indices_volt, pert=0.05, p_influence=-3., rec_power = 1.):
        v_pert = np.empty(shape=(len(voltages), len(voltages)))
        perturbing_mat = np.ones((len(voltages), len(voltages))) + pert * np.identity(len(voltages))
        v_pert[:] = np.dot(perturbing_mat, np.diag(voltages))
        #influence_mat = -np.dot(h_mat, (v_pert - voltages[:, None])).reshape(self.npix, self.npix, len(v_pert))
        influence_mat = -np.dot(h_mat, v_pert).reshape(self.npix, self.npix, len(voltages)) - rec[:, :, None]
        influence_mat = np.abs(influence_mat)
        influence_mat = np.sum(influence_mat, axis=2)
        grad_mat =  np.linalg.norm(np.gradient(rec), axis = 0)

        if np.amin(rec) <= -1.:
            rec = - rec / np.amin(rec)
        
        rec = np.log(rec + 1.0000001)
        # rec = np.abs(rec)
        # rec[rec < 0.001] = 0.001
        '''
        print('There are nans in gradient map: ', np.isnan(grad_mat).any())
        print('There are nans in influence map: ', np.isnan(influence_mat).any())
        print('There are nans in reconstruction map: ', np.isnan(rec).any())
        '''
        return grad_mat * (influence_mat) ** p_influence * rec ** rec_power, grad_mat, rec, influence_mat

    def get_next_pair(self, rec, voltages, h_mat, indices_volt, pert=0.05, cutoff=0.8, p_influence=1., rec_power = 1.):
        total_map, grad_mat, rec_log, influence_mat = np.abs(self.get_total_map(rec, voltages, h_mat, indices_volt, pert=pert, p_influence=p_influence, rec_power = rec_power))

        total_maps_along_lines = total_map[None] * self.pixel_indices
        proximity_to_boundary = np.sum(total_maps_along_lines, axis=(1, 2)) / self.sum_of_pixels_indices
        proposed_ex_line = self.voltage_all_possible[np.argsort(proximity_to_boundary)[::-1]][:10]
        '''
        plt.figure(2)
        im2 = plt.imshow(rec.astype(float), cmap=plt.cm.viridis, origin='lower', extent=[-1, 1, -1, 1])
        plt.title("reconstruction")
        plt.colorbar(im2)
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
        plt.colorbar(im6)
        plt.show()
        '''
        return proposed_ex_line, total_map


    def get_next_voltage_pairs(self, new_ex_line, total_map, number_of_voltages, counter, npix=64, cutoff=0.90):

        #region of interest index
        roi_index = total_map > cutoff * np.amax(total_map)
        weight_on_each_tri = np.sum(self.weight[roi_index], axis=0)
        new_ex_line = np.sort(new_ex_line)
        index_current = np.equal(new_ex_line[None], self.all_meas.meas[:, :2]).all(1)
        #assert np.sum(index_current) == (n_el - 2) * (n_el - 3) / 2
        jac_of_interest = self.all_meas.f.jac[index_current]
        jac_weighted = jac_of_interest * weight_on_each_tri[None, :]
        voltage_pair_initial_index = np.argsort(np.sum(jac_weighted, axis=1))[::-1]
        #print(meas[index_current][voltage_pair_initial_index])
        voltage_measurement_predictions = self.all_meas.meas[index_current, 2:][voltage_pair_initial_index]
        #print(voltage_measurement_predictions)
        return voltage_measurement_predictions[int(counter * number_of_voltages) : int((counter + 1) * number_of_voltages)]
        
    def generate_training_set(self, n_samples):
        self.measInfo = np.empty(shape=(n_samples, int((self.ne * (self.ne - 1) * (self.ne - 2) * (self.ne - 3)) / 4), 4))
        self.voltageInfo = np.empty(shape=(n_samples, int((self.ne * (self.ne - 1) * (self.ne - 2) * (self.ne - 3)) / 4)))
        self.truth_ims = np.empty(shape=(n_samples, self.npix, self.npix))

        for i in range(n_samples):
            print("Generating Anomaly:", i)
            anomaly = train.generate_anoms(2., 2.)
            self.truth_ims[i] = train.generate_examplary_output(2., self.npix, anomaly=anomaly)
            self.voltageInfo[i], self.measInfo[i] = self.get_all_voltages(anomaly)

    #def optimise(self, n_meas = 150, meas_step = 10, cutoff = 0.8, influence_mat_power = 1., pert = 0.05, rec_power = 1.):
    def optimise(self, n_meas = 500, meas_step = 50, cutoff = 0.8, influence_mat_power = 1., pert = 0.05, rec_power = 1.):
        volt_mat_start = np.empty(shape=(meas_step, 2), dtype=int)
        volt_mat_start[:, 0] = np.arange(meas_step, dtype=int) % self.ne
        volt_mat_start[:, 1] = ( np.arange(meas_step, dtype=int) + 1 ) % self.ne
        ex_mat_start = [0, self.ne // 2 ]

        loss = np.zeros((self.measInfo.shape[0], n_meas // meas_step - 1))
        num_meas = np.zeros((self.measInfo.shape[0], n_meas // meas_step - 1))
        scaled_loss = np.empty((self.measInfo.shape[0], n_meas // meas_step - 1))

        for i in range(self.measInfo.shape[0]):
            #if i>0:
            #    print("nummeas [0]", num_meas[i-1])
            meas = np.empty(shape=(meas_step, 4))
            meas[:, :2] = ex_mat_start
            meas[:, 2:] = volt_mat_start
            new_meas = np.empty(meas.shape)
            new_meas[:] = meas[:]
            q = 0
            counter = np.zeros(int(self.ne * (self.ne - 1) / 2))
            mask = np.zeros((int((self.ne * (self.ne - 1) * (self.ne - 2) * (self.ne - 3)) / 4)))
            while (meas.shape[0]) <= n_meas - meas_step:
                #t_all = time()
                mask_new = self.extract_voltages(self.measInfo[i], new_meas)
                mask = (mask + mask_new).astype(bool)
                indices_volt = np.where(mask)[0]
                
                h_mat = self.h_mat(mask)
                voltageValues = self.voltageInfo[i, mask]
                delta_v = voltageValues - self.all_meas.f.v[mask]
                rec = - np.einsum('ij, j -> i ', h_mat, delta_v, optimize='greedy').reshape((self.npix, self.npix))
                loss[i, q] = L2_Loss(rec, self.truth_ims[i])
                num_meas[i, q] = meas.shape[0]
                '''
                plt.figure(1)
                im1 = plt.imshow(self.truth_ims[i].astype(float), cmap=plt.cm.viridis, origin='lower', extent=[-1, 1, -1, 1])
                plt.title("truth")
                plt.colorbar(im1)
                '''
                sugg_ex_line, total_map = self.get_next_pair(rec, voltageValues, h_mat, indices_volt, pert=pert, cutoff=cutoff, p_influence=influence_mat_power, rec_power = rec_power)
                loc_index_src = self.ex_mat_all[:, None, 0] == sugg_ex_line[None, :, 0]
                loc_index_sink = self.ex_mat_all[:, None, 1] == sugg_ex_line[None, :, 1]
                loc = (loc_index_src * loc_index_sink).astype(bool)
                j = 0
                #t = time()
                while True:
                    masked_counter = counter[loc[:, j]]
                    if masked_counter < (self.ne - 2) * (self.ne - 3)//(2*meas_step):
                        counter[loc[:, j]] += 1
                        sugg_ex_line = sugg_ex_line[j]
                        #t2 = time()
                        new_volt_pairs = self.get_next_voltage_pairs(sugg_ex_line, total_map, meas_step, masked_counter, npix=self.npix, cutoff=cutoff)
                        #print(time() - t2, 'for voltage recommendation')
                        break
                    else:
                        j += 1
                #print('time for loop is ', time() - t)
                new_meas = np.empty((new_volt_pairs.shape[0], 4))
                new_meas[:, :2] = sugg_ex_line
                new_meas[:, 2:] = new_volt_pairs
                meas = np.concatenate((meas, new_meas), axis=0)
                #print('Time iteration ',time()-t_all)
                q += 1
                #get the suggestions and build the meas matrix
            #print('loss', loss[i])
        #gradient = (np.diff(loss/np.amin(loss, axis=1)[:, None], axis=1)/np.diff(num_meas, axis=1))
        '''
        if ((gradient < 1e-10).any()):
            #grad_index = np.where(gradient < 1e-10)
            print("hui123")
        if ((num_meas == 0).any()):
            print('hgyftrzxtdfygh')
            num_indices = np.where(num_meas == 0)
            #print("shape", loss[loss_indices[0], (n_meas//meas_step - 6):].shape)
            print("shapes of zero indices", num_indices[0].shape, num_indices[1].shape)
            print(num_meas[num_indices[0], (n_meas//meas_step - 6):])
        if ((loss < 1e-15).any()):
            print("kurec")
        '''
        #denom = (1 + np.arange(loss.shape[1])) * 10
        scaled_loss[:] = loss[:]/np.amin(loss, axis=1)[:, None]#/denom[::-1][None]

        return np.sum(scaled_loss)
    
    def objective_function(self, params):
        print(params)
        return self.optimise(n_meas = 150, meas_step = 10, cutoff = params[0], influence_mat_power = params[1], pert = params[2], rec_power = params[3])
    
def write_to_csv(filename, outputs, x_iters, hyperparameter_names, params):
    """
    ARGS: filename: Name of the csv file to write the data to,
          results: a list of differnet MSE results for all hyperparameter sets,
          x_iters: the values for each hyperparameter passed in using the 'bounds' variable,
          hyperparameter_names: A list of all the hyperparameters used,
          n_samples: number of iterations of training and testing which were then averaged,
          time: total script execution time,
          cores: number of cores used to do multiple iterations of n_samples simultaneously,
    OUTPUTS: none, writes a csv file to cwd.
    
    This function takes in all the necessary details you would want recorded to use in later analysis
    """
    if type(x_iters)==np.ndarray:
        x_iters = np.insert(x_iters,0,outputs,axis=1)
    else:
        for i in range(0,len(outputs)):
            x_iters[i].insert(0,outputs[i]) 
    
    writer = csv.writer(open(filename,'w'),lineterminator ='\n')
    writer.writerow(["Time:", "Samples","random calls","calls","n_el","n_per_el","internodal h0",
                     "a mesh coeff", "b mesh coeff", "contact width in % of side length"])
    writer.writerow([params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8],params[9]])
    writer.writerow(hyperparameter_names)
    writer.writerows(x_iters)
    print("Written to",filename,"successfully.")
    return None

n_samples = 500 # Default was 1000
n_el = 20
n_per_el=3
h0 = 0.08 # DONT FORGET TO CHANGE THE MESH PARAMS IN getJacobian() #!#!#!
a_mesh_coeff = 0.45
b_mesh_coeff = 0.3
contact_width = 5    # in percentage of side length
init_time = datetime.now()
opt = OptimiseMeasurementAlgorithm(n_samples=n_samples, ne=n_el, npix=64, n_per_el=3)
#test.optimise()
#bounds = [(0.5, 1.), (-10., 1.), (0., 1.), (-1., 10.)]
#initial = [0.97, -10., 0.5, 10.]
bounds = [(0.20, 0.99), (-40, 80), (0.0001, 1.), (-40, 80)]
initial = [[0.97, -10, 0.5, 10], [0.90, -10, 0.287, 6], [0.7,-14,0.156,24]]


           # [0.5481592754819973, 1.4176116038552067, 0.19726018135575465, 23.80169554449231],
           # [0.6578646713879979, 62.46382446984772, 0.011259519560848977, 51.74113486522847],
           # [0.7820637953529448, 38.68416844488989, 0.1443599627752602, 95.59946390653748]]
# n_rand_calls = 4
# n_calls = 10
n_rand_calls = 50
n_calls = 150
# result = gp_minimize(opt.objective_function, bounds,
#                     acq_func = "gp_hedge", n_calls=n_calls, n_random_starts=n_rand_calls, verbose=True, noise=1e-3, n_jobs=4)
result = gp_minimize(opt.objective_function, bounds, x0=initial,
                    acq_func = "gp_hedge", n_calls=n_calls, n_random_starts=n_rand_calls, verbose=True, n_jobs=4)
fin_time = datetime.now()
print("\nBayesian GP Optimisation execution time: ", (fin_time-init_time), "(hrs:mins:secs)")
print('Results: Output: %.4f, cutoff: %.6f, influence matrix power: %d,  perturb coeff: %.6f, rec power: %.6f' % (result.fun,
                              result.x[0], result.x[1],result.x[2], result.x[3]))
hyperparams = ["Output","cutoff","influence matrix power","perturbation coefficient","rec power"]
filename = 'Bayesian_adaptiveESA_5'+'.csv'
run_params = [fin_time-init_time, n_samples, n_rand_calls, n_calls, n_el, n_per_el, h0, a_mesh_coeff, b_mesh_coeff, contact_width]
write_to_csv(filename, result.func_vals, result.x_iters, hyperparams, run_params)
#dump(result, "result.pkl")

'''
anom =  train.generate_anoms(2., 2.)
_, meass = test.get_all_voltages(anom)
print(meass)



measurements = np.random.randint(0, 20, (20, 4))
print(measurements)
all_meas = getJacobian()
w_mat = getWeightSigmoid(all_meas.mesh_obj['node'], all_meas.mesh_obj['element'])
t = time()
print(h_mat(measurements, all_meas, w_mat=w_mat).shape)
print(time()-t)
'''