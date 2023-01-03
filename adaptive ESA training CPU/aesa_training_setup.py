# -*- coding: utf-8 -*-
"""
Created on Thu Nov 5 10:04:19 2020
Authors: Ivo Mihov, Vasil Aramov, Adam Coxson and Frederik Brookebarnes,
MPhys Undergrads, The University of Manchester
Project: Automated Electrical Impedance Tomography of Graphene
Module: pretrain_AdaptiveESA_demo.py
Dependancies: 
"""
import numpy as np
import greit_rec_training_set as train
from pyeit.eit.fem_forallmeas import Forward
from pyeit.eit.fem_for_given_meas import Forward as FEM
from pyeit.eit.utils import eit_scan_lines
from meshing import mesh

from time import time
from datetime import datetime
import h5py as h5
import sys
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pickle     # For saving/loading anomaly classes

from measurement_optimizer import *

def saveJacobian(filename='forward__.h5',n_el=20, n_per_el=3):
    # number electrodes
    el_pos = np.arange(n_el * n_per_el)
    # create an object with the meshing characteristics to initialise a Forward object
    edge = 0.08
    el_width = 0.04
    mesh_params = [0.045, 2000, 3.67, 3.13]
    mesh_obj = mesh(n_el=n_el, num_per_el=n_per_el, edge=edge, el_width=el_width, mesh_params=mesh_params) 

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
    h = h5.File(filename, 'w')

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
    greit.setup(p=0.2, lamb=0.01, n=n_pix)
    rel_perm = greit.solve(f_anom.v, f.v)
    rel_perm = rel_perm.reshape((n_pix, n_pix)) # from (n_pix, n_pix, 1) to (n_pix, n_pix)

    return rel_perm

def lossL2(im, true):
    return np.sum((im - true)**2)

def testAlgorithm(fileJac, anomaly=None, n=4, a=2., ne=20, pert=0.5, cutoff=0.97, p_influence=-10., p_rec=10.):
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
    #anomaly = None
    if anomaly is None:
        anomaly = train.generate_anoms(a, a)
    #print(anomaly)
    ex_mat_all = train.generateExMat(ne=ne)
    volt_mat_all = train.generateExMat(ne=ne)
    index_1 = ((np.absolute(volt_mat_all[:,1] - volt_mat_all[:,0]) == 1) + (np.absolute(volt_mat_all[:,1] - volt_mat_all[:,0]) == 19))
    volt_mat_1 = volt_mat_all[index_1]
    volt_mat_1 = volt_mat_1[np.argsort(volt_mat_1[:, 1], axis=0)]
    volt_mat_1 = volt_mat_1[np.argsort(volt_mat_1[:, 0]%5, axis=0)]
    
    # opp-adj
    index_stand = ((np.absolute(meas[:,1] - meas[:,0]) == 10))# + (np.absolute(meas[:,1] - meas[:,0]) == 19))
    index_stand *= ((np.absolute(meas[:,3] - meas[:,2]) == 1) + (np.absolute(meas[:,3] - meas[:,2]) == 19))
    
    #print(np.sum(index_stand))
    meas_11 = meas[index_stand]
    ind = np.argsort(meas_11[:, 1], axis=0)
    meas_11 = meas_11[ind]
    ordered = meas_11[:10]
    suggested = ordered[:]
    #print(ordered)
    
    # adj-adj
    index_stand2 = ((np.absolute(meas[:,1] - meas[:,0]) == 1))# + (np.absolute(meas[:,1] - meas[:,0]) == 19))
    index_stand2 *= ((np.absolute(meas[:,3] - meas[:,2]) == 1) + (np.absolute(meas[:,3] - meas[:,2]) == 19))
    
    meas_2 = meas[index_stand2]
    ind2 = np.argsort(meas_2[:, 1], axis=0)
    meas_2 = meas_2[ind2]
    ordered2 = meas_2[:10]
    suggested2 = ordered2[:]
    #print(ordered2)

    counter = np.zeros(190)
    numS = []
    numO = []
    numO2 = []
    lossS = []
    lossO = []
    lossO2 = []
    for i in range(int((len(meas_11)-10)//n)):
        ex_pred, recSugg, trSugg, numSugg, total_map = findNextPair(fileJac, ne, anomaly=anomaly, meas=suggested, a=2., npix=64, pert=pert, p_influence=p_influence, p_rec=p_rec)

        recOrd, _, _, _, trOrd, numOrd = simulateMeasurements(fileJac, anomaly=anomaly, measurements=ordered, n_el=ne, n_pix=64, a=a)
        recOrd2, _, _, _, trOrd2, numOrd2 = simulateMeasurements(fileJac, anomaly=anomaly, measurements=ordered2, n_el=ne, n_pix=64, a=a)

        lossSugg = lossL2(recSugg, trSugg)
        lossOrd = lossL2(recOrd, trOrd)
        lossOrd2 = lossL2(recOrd2, trOrd2)
        
        numS.append(numSugg)
        numO.append(numOrd)
        numO2.append(numOrd2)
        lossS.append(lossSugg)
        lossO.append(lossOrd)
        lossO2.append(lossOrd2)

        # print('Reconstructing with', 10 + i * n, 'measurements:')
        # print('Loss Adaptive ESA =', lossSugg)
        # print('Loss Ordinary Opp-adj =', lossOrd)
        # print('Loss Ordinary Adj-adj =', lossOrd2)

        rem_r = (np.sum((volt_mat_1 != ex_pred[0]) + (volt_mat_1 != ex_pred[1]), axis=1)).astype(bool)
        volt_mat = volt_mat_1[rem_r]
        j = 0
        while True:
            loc = (np.sum((ex_mat_all == ex_pred[None, j]), axis=1) == 2).astype(bool)
            if counter[loc] == 0 or counter[loc] < (ne - 2) * (ne - 3)//(5*n):
                counter[loc] += 1
                ex_pred = ex_pred[j]
                new_volt_meas = findNextVoltagePair(ex_pred, fileJac, total_map, n, counter[loc], npix=64, cutoff=cutoff)
                #print(new_volt_meas)
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
        ordered2 = meas_2[:int(10 + n * (i+1))]
        suggested = np.concatenate((suggested, x), axis=0)

    
    # fig1, ax1 = plt.subplots()
    # ax1.plot(numS, lossS, 'rx')
    # ax1.set_title('L2 Loss Function of Algorithm Selected Measurements')
    # ax1.set_xlabel('Num. of Measurements')
    # ax1.set_ylabel('L2 Loss Squared')

    # fig2, ax2 = plt.subplots()
    # ax2.plot(numO, lossO, 'bx')
    # ax2.set_title('L2 Loss Function of a Commonly Used Measurement Technique')
    # ax2.set_xlabel('Num. of Test Sample')
    # ax2.set_ylabel('L2 Loss Squared')
    # plt.show()
    parameterS = np.gradient(lossS) / np.gradient(numS)
    parameterS /= np.array(numS)**1.
    parameterS = np.mean(parameterS)

    parameterO = np.gradient(lossO) / np.gradient(numO)
    parameterO /= np.array(numO)**1.
    parameterO = np.mean(parameterO)
    
    parameterO2 = np.gradient(lossO2) / np.gradient(numO2)
    parameterO2 /= np.array(numO2)**1.
    parameterO2 = np.mean(parameterO2)
    #parameterO2=1.0
    return np.array(lossS), np.array(numS), np.array(lossO), np.array(numO), np.array(lossO2), np.array(numO2), parameterS/parameterO, parameterS/parameterO2


# saveJacobian(n_el=20, n_per_el=3)
# testAlgorithm('forward.h5', n=5)



# measurements_0 = cp.random.randint(0, 20, (1500, 4))

# while cp.sum(cp.equal(measurements_0[:, :, None], measurements_0[:, None, :])) > 4 * len(measurements_0):
#     measurements_0 = cp.random.randint(0, 20, (35, 4))
# v_meas = 10 * np.ones(len(measurements_0)) * np.random.randn()

#print("\n --- Saving Jacobian --- \n")
#saveJacobian()
#print("\n --- Finding next pair --- \n")
#findNextPair('forward.h5', 20, el_dist=1)

#saveJacobian()
#print("\n --- Optimising the batch --- \n")
#batchOptimize('forward.h5', batchSize=90)
#print("\n --- Testing the algorithm Uwu --- \n")
#testAlgorithm('forward.h5', n=5)


#findNextPair('forward__.h5', 20, el_dist=1)

#print("\n --- Optimising the batch --- \n")
# batchOptimize('forward_.h5', batchSize=90)
# #print("\n --- Testing the algorithm Uwu --- \n")
# testAlgorithm('forward_.h5', n=5)

# anomaly_list=np.zeros(10).astype(object)
# for i in range(0,200):
#     anomaly = train.generate_anoms(2.0, 2.0)
#     anomaly_list[i]=anomaly
# save_name = 'anoms_evaluation'
# with open(str(save_name)+'.pkl','wb') as save_file:
#             pickle.dump(anomaly_list, save_file)
        

#load_name = 'anoms_8'
load_name = 'anoms_evaluation'
try:
    with open(str(load_name)+'.pkl','rb') as load_file:
        anomaly_list = pickle.load(load_file)
except:
    print("Saved file not found")


param_list = [[0.97,-10,0.5,10],
              [0.6357154598817543,-8.74675301016891,0.3141784084620273,0.5546683314342027],
              [0.899402163, 9.697604843, 0.28658324, -6.184714296],
              [0.625,84,0.133,75],
              [0.343,-9.0,0.35,8.0],
              [0.814056685,-8.968430304,0.736364284,7.307916787],
              [0.69037305,-14,0.156208803,24] # decent - even better [0.7,-14,0.156,24]
              ]
param_list = [[0.97,-10,0.5,10],
              [0.6357154598817543,-8.74675301016891,0.3141784084620273,0.5546683314342027],
              [0.90, -9.7, 0.287, 6.2], # pretty good
              [0.625,84,0.133,75],
              [0.343,-9.0,0.35,8.0],
              [0.82,-9.0,0.74,7.3], # reasonable - close
              [0.579799601,0,0.01595223,26],
              [0.7,-14,0.156,24],
              [0.95,-20,0.718,22],
              [0.954,1,0.316,11],
              [0.941295,5,0.303189,-14]
              ] 
param_list = [[0.97,-10,0.5,10],[0.8,-15,0.2,20],[0.7,-14,0.156,24],
              [0.69037305,-14,0.156208803,24],[0.82,-9.0,0.74,7.3],
              [0.778696654,39,0.308756905,40],[0.606880753,60,0.483382399,39],
              [0.80460672,38,0.361389917,41],[0.93971779,79,0.403593986,-1]]
#param_list = [[0.502402696,46,0.67380612,-21], [0.606880753,60,0.483382399,39]]
#params = param_list[0]
n_repeats=3
iterations = 200
sum_list_aesa = np.zeros(n_repeats)
sum_list_oppadj = np.zeros(n_repeats)


#for k in range(5,9):
# for k in [0,3,7]:
#     params=param_list[k]
#     start = datetime.now()
#     for j in range(0,n_repeats):
#         filename="jac_"+str(j+3)+".h5"
#         #filename="jac_"+str(j+1)+".h5"
#         #print(filename)
#         lossS, numS = [], []
#         lossO, numO = [], []
#         lossO2, numO2 = [], []
#         for i in range(iterations):
#             lossSugg, numSugg, lossOrd, numOrd, lossOrd2, numOrd2, _, _ = testAlgorithm(filename, anomaly=anomaly_list[i], n=10, cutoff=params[0], p_influence=params[1],pert=params[2],p_rec=params[3])
        
#             minim = np.amin(lossSugg[:12])
#             #print('minimal of', i, '=', minim)
            
#             lossOrd = lossOrd[numOrd <= np.amax(numSugg)]
#             numOrd = numOrd[numOrd <= np.amax(numSugg)]
#             # lossOrd2 = lossOrd2[numOrd2 <= np.amax(numSugg)]
#             # numOrd2 = numOrd2[numOrd2 <= np.amax(numSugg)]
#             lossS.append(lossSugg[:12]/minim)
#             numS.append(numSugg[:12])
#             lossO.append(lossOrd[:12]/minim)
#             numO.append(numOrd[:12])
#             # lossO2.append(lossOrd2[:12]/minim)
#             # numO2.append(numOrd2[:12])
            
        
        
#         lossO = np.array(lossO)
#         #lossO2 = np.array(lossO2)
#         lossS = np.array(lossS)
#         lossO_sum = np.sum(lossO, axis=0)
#         # lossO2_sum = np.sum(lossO2, axis=0)
#         lossS_sum = np.sum(lossS, axis=0)
#         #print('lossO' , lossO_sum)
#         #print('lossO2' , lossO2_sum)
#         #print('lossS' , lossS_sum)
#         sum_list_aesa[j] = lossS_sum[-1]
#         sum_list_oppadj[j] = lossO_sum[-1]
#         print('Final summed loss AESA:', lossS_sum[-1])
#         print('Final summed loss opp-adj:', lossO_sum[-1])
#         #print("params:",params)
        
    
#     fin = datetime.now()
#     print("Time:",fin-start)
    
#     aesa_sum_loss = np.mean(sum_list_aesa)
#     aesa_sum_loss_err = np.std(sum_list_aesa)
#     oppadj_sum_loss = np.mean(sum_list_oppadj)
#     oppadj_sum_loss_err = np.std(sum_list_oppadj)
#     print("Params:",params, "Samples:",iterations, "Repeats:",n_repeats)
#     print("AESA:",aesa_sum_loss/iterations,"\u00B1",aesa_sum_loss_err/iterations)
#     print("Opp-Adj:",oppadj_sum_loss/iterations,"\u00B1",oppadj_sum_loss_err/iterations)


params=param_list[0]
start = datetime.now()
filename="jac_2.h5"
lossS, numS = [], []
lossO, numO = [], []
lossO2, numO2 = [], []
for i in range(8):
    lossSugg, numSugg, lossOrd, numOrd, lossOrd2, numOrd2, _, _ = testAlgorithm(filename, anomaly=anomaly_list[i], n=10, cutoff=params[0], p_influence=params[1],pert=params[2],p_rec=params[3])

    minim = np.amin(lossSugg[:12])
    #print('minimal of', i, '=', minim)
    
    lossOrd = lossOrd[numOrd <= np.amax(numSugg)]
    numOrd = numOrd[numOrd <= np.amax(numSugg)]
    lossOrd2 = lossOrd2[numOrd2 <= np.amax(numSugg)]
    numOrd2 = numOrd2[numOrd2 <= np.amax(numSugg)]
    lossS.append(lossSugg[:12]/minim)
    numS.append(numSugg[:12])
    lossO.append(lossOrd[:12]/minim)
    numO.append(numOrd[:12])
    lossO2.append(lossOrd2[:12]/minim)
    numO2.append(numOrd2[:12])
    


lossO = np.array(lossO)
lossO2 = np.array(lossO2)
lossS = np.array(lossS)
lossO_sum = np.sum(lossO, axis=0)
lossO2_sum = np.sum(lossO2, axis=0)
lossS_sum = np.sum(lossS, axis=0)
#print('lossO' , lossO_sum)
#print('lossO2' , lossO2_sum)
#print('lossS' , lossS_sum)
print("params:",params)
print('Final summed loss AESA:', lossS_sum[-1])
print('Final summed loss opp-adj:', lossO_sum[-1])
print('Final summed loss adj-adj:', lossO2_sum[-1])
fin = datetime.now()
print("Time:",fin-start)
    



#plt.close('all')
for i in range(8):
    plt.figure()
    plt.plot(numS[i], lossS[i], 'r.',label='A-ESA')
    plt.plot(numO[i], lossO[i], 'g.',label='Opp-Adj')
    plt.plot(numO2[i], lossO2[i], 'b.',label='Adj-Adj')
    plt.xlabel('Number of measurements')
    plt.ylabel('Loss')
    plt.grid(b=True, which='major', axis='both')
    plt.legend()
    plt.minorticks_on()
    plt.show()


fig1, ax1 = plt.subplots()
ax1.plot(numS[0], lossS[0], 'r.')
ax1.plot(numS[1], lossS[1], 'b.')
ax1.plot(numS[2], lossS[2], 'g.')
ax1.plot(numS[3], lossS[3], 'm.')
ax1.plot(numS[4], lossS[4], 'c.')

#ax1.set_title('L2 Loss of Our Algorithm Selected Measurement Technique')
#ax1.set_title('A-ESA')
ax1.set_xlabel('Number of measurements')
ax1.set_ylabel('Loss')
plt.grid(b=True, which='major', axis='both')
plt.minorticks_on()


fig2, ax2 = plt.subplots()
ax2.plot(numO[0], lossO[0], 'r.')
ax2.plot(numO[1], lossO[1], 'b.')
ax2.plot(numO[2], lossO[2], 'g.')
ax2.plot(numO[3], lossO[3], 'm.')
ax2.plot(numO[4], lossO[4], 'c.')
#ax2.set_title('L2 Loss of a Common Measurement Technique: El. Dist. = 10, Step = 1')
#ax2.set_xlabel('Number of Measurements')
#ax2.set_title('Opp-Adj')
ax2.set_xlabel('Number of measurements')
ax2.set_ylabel('Loss')
plt.grid(b=True, which='major', axis='both')
plt.minorticks_on()

fig3, ax3 = plt.subplots()
ax3.plot(numO2[0], lossO2[0], 'r.')
ax3.plot(numO2[1], lossO2[1], 'b.')
ax3.plot(numO2[2], lossO2[2], 'g.')
ax3.plot(numO2[3], lossO2[3], 'm.')
ax3.plot(numO2[4], lossO2[4], 'c.')
#ax2.set_title('L2 Loss of a Common Measurement Technique: El. Dist. = 10, Step = 1')
#ax2.set_xlabel('Number of Measurements')
#ax2.set_title('L2 Loss of Common Measurement Pattern (opposite)')
#ax3.set_title('Adj-Adj')
ax3.set_xlabel('Number of measurements')
ax3.set_ylabel('Loss')
plt.grid(b=True, which='major', axis='both')
plt.minorticks_on()
plt.show()



# perm = solveGREIT()

# fig, ax = plt.subplots(figsize=(6, 4))
# im = ax.imshow(np.real(perm), interpolation='none', cmap=plt.cm.viridis, origin='lower', extent=[-1,1,-1,1])
# fig.colorbar(im)
# ax.axis('equal')

# plt.show()
