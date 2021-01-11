# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 12:13:08 2020
Author: Adam Coxson, MPhys Undergrad, The University of Manchester
Project: Automated Electrical Impedance Tomography for Graphene
Module: get_next_prediction.py
Dependancies: pyeit.eit: .fem_for_given_meas .fem_forallmeas .utils, greit_rec_training_set, measurement_optimizer
    
This script contains functions which can call the relevant modules to perform electrical impedance tomography
using a variety of electrode selection algorithms. This creates arrays formatted for four-terminal measurements as 
(I+,I-,V+,V-). This can either simulate the voltage values by synthetic anomaly insertion onto pyEIT delaunay triangle
meshes or can be substituded for real voltages measurements. 

This script has been created to work with component_control.py and selection_algoritms.py created by the authors to 
automate hardware for EIT measurements.
See very bottom of this script for some info on the required format for the adaptive ESA to iterate.

"""
# Generic package imports
import os
import numpy as np
import h5py as h5 # for file saving
import cupy as cp # CUDA GPU accelerated python
import matplotlib.pyplot as plt
# Modules
import measurement_optimizer as measopt # contains functions for the adaptive ESA.
import greit_rec_training_set as train
from pyeit.eit.fem_forallmeas import Forward
from pyeit.eit.fem_for_given_meas import Forward as Forward_given
from pyeit.eit.utils import eit_scan_lines
from meshing import mesh

def save_small_Jacobian(save_filename, n_el=20, n_per_el=3):
    # number electrodes
    el_pos = np.arange(n_el * n_per_el)
    # create an object with the meshing characteristics to initialise a Forward object
    mesh_obj = mesh(n_el)
    ex_mat = train.orderedExMat(n_el=20)
    #ex_mat = train.generateExMat(ne=n_el)
    fwd = Forward_given(mesh_obj, el_pos, n_el)

    f, meas, new_ind = fwd.solve_eit(ex_mat=ex_mat, perm=fwd.tri_perm)
    
    #print(f)
    ind = np.arange(len(meas))
    np.random.shuffle(ind)
    pde_result = train.namedtuple("pde_result", ['jac', 'v', 'b_matrix'])

    f = pde_result(jac=f.jac[ind], v=f.v[ind], b_matrix=f.b_matrix[ind])
    meas = meas[ind]
    new_ind = new_ind[ind]
    h = h5.File(save_filename, 'w')

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
    
def saveJacobian(save_filename, n_el=20, n_per_el=3):
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

    f = pde_result(jac=f.jac[ind], v=f.v[ind], b_matrix=f.b_matrix[ind])
    meas = meas[ind]
    new_ind = new_ind[ind]
    h = h5.File(save_filename, 'w')

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

def getNextPrediction(fileJac: str, measuring_electrodes: np.ndarray, voltages: np.ndarray, 
              num_returned: int=10, n_el: int=20, n_per_el: int=3, n_pix: int=64, pert: float=0.5, 
              p_influence: float=-10., p_rec: float=10., p: float=0.2, lamb:float=0.1) -> np.ndarray:
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
        'node':    p,
        'perm':    perm}
    # list all possible active/measuring electrode permutations of this measurement
    meas = cp.array(meas)
    # find their indices in the already calculated const. permitivity Jacobian (CPJ)
    measuring_electrodes = cp.array(measuring_electrodes)
    measurements_0 = cp.amin(measuring_electrodes[:, :2], axis=1)
    measurements_1 = cp.amax(measuring_electrodes[:, :2], axis=1)
    measurements_2 = cp.amin(measuring_electrodes[:, 2:], axis=1)
    measurements_3 = cp.amax(measuring_electrodes[:, 2:], axis=1)
    measuring_electrodes = cp.empty((len(measuring_electrodes), 4))
    measuring_electrodes[:, 0] = measurements_0
    measuring_electrodes[:, 1] = measurements_1
    measuring_electrodes[:, 2] = measurements_2
    measuring_electrodes[:, 3] = measurements_3
    index = (cp.sum(cp.equal(measuring_electrodes[:, None, :], meas[None, :, :]), axis=2) == 4)
    index = cp.where(index)
    #print(index)
    ind = cp.unique(index[1])
    #print(ind)
    i = cp.asnumpy(ind)
    j = index[0]
    mask = np.zeros(len(meas), dtype=int)
    mask[i] = 1
    mask = mask.astype(bool)
    # take a slice of Jacobian, voltage readings and B matrix (the one corresponding to the performed measurements)
    file = h5.File(fileJac, 'r')
    jac = file['jac'][mask, :][()]
    v = file['v'][mask][()]
    b = file['b'][mask, :][()]
    file.close()
    # put them in the form desired by the GREIT function
    pde_result = train.namedtuple("pde_result", ['jac', 'v', 'b_matrix'])
    f = pde_result(jac=jac,
           v=v,
           b_matrix=b)
    
    # now we can use the real voltage readings and the GREIT algorithm to reconstruct
    greit = train.greit.GREIT(mesh_obj, el_pos, f=f, ex_mat=(meas[index[1], :2]), step=None)
    greit.setup(p=p, lamb=lamb, n=n_pix)
    h_mat = greit.H
    reconstruction = greit.solve(voltages, f.v).reshape(n_pix, n_pix)
    # fix_electrodes_multiple is in meshing.py
    _, el_coords = train.fix_electrodes_multiple(centre=None, edgeX=0.1, edgeY=0.1, a=2, b=2, ppl=n_el, el_width=0.02, num_per_el=3)
    # find the distances between each existing electrode pair and the pixels lying on the liine that connects them
    pixel_indices, voltage_all_possible = measopt.find_all_distances(reconstruction, h_mat, el_coords, n_el, cutoff=0.8)
    # call function get_total_map that generates the influence map, the gradient map and the log-reconstruction
    total_map, grad_mat, rec_log = np.abs(measopt.get_total_map(reconstruction, voltages, h_mat, pert=pert, p_influence=p_influence, p_rec=p_rec))
    # get the indices of the total map along the lines connecting each possible electrode pair
    total_maps_along_lines = total_map[None] * pixel_indices
    # find how close each connecting line passes to the boundary of an anomaly (where gradient supposed to be higher)
    proximity_to_boundary = np.sum(total_maps_along_lines, axis=(1, 2)) / np.sum(pixel_indices, axis=(1, 2))
    # rate the possible src-sink pairs by their proximity to existing anomalies
    proposed_ex_line = voltage_all_possible[np.argsort(proximity_to_boundary)[::-1]][:num_returned]

    number_of_voltages = 10
    # generate the voltage measuring electrodes for this current driver pair
    proposed_voltage_pairs = measopt.findNextVoltagePair(proposed_ex_line[0], fileJac, total_map, number_of_voltages, 0, npix=n_pix, cutoff=0.97)
    return proposed_ex_line, proposed_voltage_pairs, reconstruction, total_map

    
def current_matrix_generator(n_el: int, ex_mat_length: int, ex_pair_mode='adj', spaced: bool=False):
    """
    This function generates an excitation matrix of current pairs using the adj method, opposite method, all 
    possible pairs or custom step. It can iterate and build the matrix contiguously, assigning electrodes in 
    turn around the perimeter or 'spaced' can be set to True, which spaces out the measurements slightly.
    I.e. for the adj method, spaced=True might give [(1,2),(5,6),(10,11)] as opposed to [(1,2),(2,3),(3,4)].
    To Do:
        - Add in a randomly assigned current pairs method
    
    Parameters
    ----------
    n_el : int
        The number of electrodes.
    ex_mat_length : int
        The desired length of the excitation matrix, if none will generate maximum number of measurements
        for the given ex_pair_mode.
    ex_pair_mode : int or string 
        The default is 'adj' or step of 1, but can be 'opp' step of n_el/2 or 'all'. Also accepts a
        custom step value. 
    spaced : bool, optional
        If True, this will activate orderedExMat which spaces out measurements around the perimeter.

    Returns
    -------
    ex_mat : ndarray (N, 2)
        List of the current pairs (Source, sink).
    """
    
    if (ex_pair_mode == 1 or ex_pair_mode == 'adj'):
        print("Adjacent current pairs mode")
        dist = 1
    elif (ex_pair_mode == n_el//2 or ex_pair_mode == 'opp'):
        print("Opposite current pairs mode")
        dist = n_el//2
    elif (ex_pair_mode == 'all'):
        print("All current pairs mode")
    elif (ex_pair_mode != 1 and ex_pair_mode != n_el//2 and (1 < ex_pair_mode < n_el-1)):
        print("Custom distance between current pairs. Dist:", ex_pair_mode)
        dist = ex_pair_mode
    else:
        print("Incorrect current pair mode selected. Function returning 0.")
        return 0
    if (ex_pair_mode == 'all'):
        ex_mat = train.generateExMat(ne=n_el)
    elif spaced == True:
        ex_mat = train.orderedExMat(n_el=n_el, el_dist=dist)
    elif spaced == False:
        ex_mat = eit_scan_lines(ne=n_el, dist=dist)
    else:
        print("Something went wrong ...")
        
    if ex_mat_length is not None:
        ex_mat = ex_mat[0:ex_mat_length]
    return ex_mat
    
def volt_matrix_generator(ex_mat: np.ndarray, n_el: int, volt_pair_mode: str='adj'):
    """
    This function generates an excitation matrix of voltage pairs using the adj method, opposite method, all 
    possible pairs or custom step. 
    To Do:
        - Add in a randomly assigned voltage pairs method

    Parameters
    ----------
    ex_mat : np.ndarray (N, 2)
        List of the current pairs (Source, sink).
    n_el : int
        The number of electrodes.
    volt_pair_mode : str, optional
        Defines the voltage pair assigment mode. Can be 'adj', 'opp', 'all', or a custom step. 
        1 and n_el/2 also works for adj and opp method. The default is 'adj'.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if (volt_pair_mode == 'all'):
        print("All valid voltage pairs mode")
        all_pairs = cp.asnumpy(measopt.volt_mat_all(n_el))
        volt_mat = []
        for ex_line in ex_mat: # Each ex_line is a current pair. Removing volt pairs which cross over with the current pair
            valid_pairs = [pair for pair in all_pairs if pair[0]!=ex_line[0] if pair[0]!=ex_line[1] if pair[1]!=ex_line[0] if pair[1]!=ex_line[1]]
            valid_pairs = np.array(valid_pairs)
            #("No. valid pairs: ", len(valid_pairs))
            if len(volt_mat)==0:
                volt_mat = valid_pairs
            else:
                volt_mat = np.append(volt_mat,valid_pairs,axis=0)
        ind = np.zeros(len(volt_mat))
        pairs_per_current = int(len(volt_mat)/len(ex_mat))
        for i in range(0,len(ex_mat)):
            ind[pairs_per_current*i:pairs_per_current*(i+1)] = i
            
    elif (volt_pair_mode == 1 or volt_pair_mode == 'adj'):
        print("Adjacent voltage pairs mode")
        volt_mat, new_ex_mat, new_ind = measopt.voltMeterwStep(n_el, cp.asarray(ex_mat), step_arr=1, parser=None)
        ind = cp.asnumpy(new_ind)
        volt_mat = cp.asnumpy(volt_mat)
        
    elif (volt_pair_mode == n_el//2 or volt_pair_mode == 'opp'):
        print("Opposite voltage pairs mode")
        volt_mat, new_ex_mat, new_ind = measopt.voltMeterwStep(n_el, cp.asarray(ex_mat), step_arr=n_el//2, parser=None)
        ind = cp.asnumpy(new_ind)
        volt_mat = cp.asnumpy(volt_mat)
        
    elif (volt_pair_mode != 1 and volt_pair_mode != n_el//2 and (1 < volt_pair_mode < n_el-1)):
        print("Custom distance between voltage pairs. Dist:", volt_pair_mode)
        volt_mat, new_ex_mat, new_ind = measopt.voltMeterwStep(n_el, cp.asarray(ex_mat), step_arr=volt_pair_mode, parser=None)
        ind = cp.asnumpy(new_ind)
        volt_mat = cp.asnumpy(volt_mat)
        
    else:
        print("Incorrect voltage pair mode selected")
        

    return volt_mat, ind.astype(np.int32)

def initialise_ex_volt_mat(current_mode:str='adj', volt_mode:str='adj', n_el:int=32, ex_mat_length:int=10, ex_spaced:bool=False):
    """
    This function is used to initialise a current and voltage excitation matrix which can be passed into measurement hardware to
    obtain corresponding voltages or into forward.solve_eit with a synthetic anomaly mesh to simulate measurements. Default 
    generation of the ex_volt_mat is adj-adj.

    Parameters
    ----------
    current_mode : str, optional
        Defines the current pair assigment mode. Can be 'adj', 'opp', 'all', or a custom step. 
        1 and n_el/2 also works for adj and opp method. The default is 'adj'.
    volt_mode : str, optional
        Defines the voltage pair assigment mode. Can be 'adj', 'opp', 'all', or a custom step. 
        1 and n_el/2 also works for adj and opp method. The default is 'adj'. The default is 'adj'.
    n_el : int, optional
        The number of electrodes. The default is 32.
    ex_mat_length : int, optional
        The desired length of the excitation matrix, if none will generate maximum number of measurements
        for the given ex_pair_mode. The default is 10.
    ex_spaced : bool, optional
        If True, this will activate orderedExMat which spaces out current measurement pairs around the perimeter.
        See the definition for current_matrix_generator.

    Returns
    -------
    ex_volt_mat : np.ndarray (M,4) with each row [current source, current sink, voltage+, voltage-)
        The concatenated current pair and voltage pair matrix.
    ex_mat : np.ndarray (N, 2)
        List of the current pairs (Source, sink).
    volt_mat : np.ndarray (M, 2)
        List of the voltages pairs (Source, sink).
    ind : np.array of ints
        Can be used to duplicate and extend the current pair array ex_mat to match the format of volt_mat for concatenation.
        Call it like extended_ex_mat = ex_mat[ind].
    """
    ex_mat = current_matrix_generator(n_el=n_el, ex_mat_length=ex_mat_length, ex_pair_mode=current_mode, spaced=ex_spaced)
    volt_mat, ind = volt_matrix_generator(ex_mat=ex_mat, n_el=n_el, volt_pair_mode=volt_mode)
    ex_volt_mat = np.concatenate((ex_mat[ind], volt_mat), axis=1) 
    return ex_volt_mat, ex_mat, volt_mat, ind    

#def find_current_pairs()
def ex_mat_ind_finder(ex_volt_mat):
    ordered_ind = np.zeros(len(ex_volt_mat))
    previous_line = ex_volt_mat[0][0:2]
    ex_mat = previous_line
    j=0
    for i in range(0,len(ex_volt_mat)):
        same_line = ((previous_line) == (ex_volt_mat[i][0:2])) # Comparing previous ex pair to current ex_pair
        same_line = bool(same_line[0]*same_line[1]) # Only True if previous line == Current line
        if same_line is False:
            ex_mat = np.vstack((ex_mat, ex_volt_mat[i][0:2]))
            j=j+1
        ordered_ind[i]=j
        previous_line = ex_volt_mat[i][0:2]
        
    return ex_mat.astype(np.int32), ordered_ind.astype(np.int32)

def reconstruction_plot(ex_volt_mat, measured_voltages, mesh_obj=None, start_pos='left', n_el=32, p=0.2, lamb=0.01, n_pix=64):
    n_per_el = 3
    #extended_ex_mat   = ex_volt_mat[:, :2]
    volt_mat = ex_volt_mat[:, 2:]
    ex_mat, ind = ex_mat_ind_finder(ex_volt_mat)
    if mesh_obj == None:
        mesh_obj = mesh(n_el=n_el, num_per_el=n_per_el, start_pos=start_pos) # Making an empty mesh
    el_pos = np.arange(n_el * n_per_el).astype(np.int16)
    #el_pos = np.arange(n_el).astype(np.int16)
    fwd = Forward_given(mesh_obj, el_pos, n_el)
    f, meas, new_ind = fwd.solve_eit(volt_mat=volt_mat, ex_mat=ex_mat, new_ind=ind)
    greit = train.greit.GREIT(mesh_obj, el_pos, f=f, ex_mat=ex_mat)
    greit.setup(p=p, lamb=lamb, n=n_pix)
    reconstruction = greit.solve(measured_voltages, f.v, normalize=False).reshape(n_pix, n_pix)
    
    plt.figure()
    im = plt.imshow((reconstruction), cmap=plt.cm.viridis, origin='lower', extent=[-1, 1, -1, 1])
    #plt.title("GREIT Reconstruction\n(Current, Voltage) pairs: ("+str(ind[-1]+1)+", "+str(len(volt_mat))+")")
    cb = plt.colorbar(im)
    cb.ax.tick_params(labelsize=15)
    plt.yticks(ticks=[-1.0,-0.75,-0.50,-0.25,0,0.25,0.50,0.75, 1.00], labels=[-1.0,-0.75,-0.50,-0.25,0,0.25,0.50,0.75, 1.00], fontsize=15)
    plt.xticks(ticks=[-1.0,-0.75,-0.50,-0.25,0,0.25,0.50,0.75, 1.00], labels=[-1.0,-0.75,-0.50,-0.25,0,0.25,0.50,0.75, 1.00], fontsize=15)
    #plt.xlabel(r'x' , fontsize = 12, fontname = 'cmr10')
    #plt.ylabel(r'y', fontsize = 12)
    #plt.savefig(filepath+"ESA reconstruction "+str(i))
    plt.show()
    return 0 

def propose_ex_volt_mat(proposed_ex_lines, ex_volt_mat, save_filename, total_map, n_of_voltages=10, counter=0, n_pix=64):
    """
    Currently this does not properly account for repeated excitation lines, this is done outside of this function through
    use of the repeated_ex variable. However, this function does ensure that any new proposed current-voltage pairs are 
    unique, ignoring any repeated measurements. This can cause the size of the output array to vary, it is not always
    equal to n_current_pairs * N_voltages_per_current_pair, since when deleted, they are not replaced. This because more 
    apparent over multiple iterations of the ESA
    
    To Do:
        - Figure out what the counter variable does - think its to do with repeated voltages
        - Account for varying size as voltages are repeated, need to reduce the often varied size of proposed_ex_lines

    Parameters
    ----------
    proposed_ex_lines : int np.ndarray (M,2) with each row [current source, current sink)
        The current pair matrix of new proposed current pairs obtained from analysing the total map.
    ex_volt_mat : np.ndarray (M,4) with each row [current source, current sink, voltage+, voltage-)
        The previous concatenated current pair and voltage pair matrix.
    save_filename : string
        Filename of the temporary Jacobian for all elements of the proposed current pairs.
    total_map : (n_pixels, n_pixels) array of float
        Contains the total map values for each pixel of the GREIT reconstruction, see adaptive ESA theory
    n_of_voltages : int
        number of voltages per current pair to propose. The default is 10.
    counter : int, optional
        undetermined usage. The default is 0.
    npix : int, optional
        The number of pixels to be used for the GREIT reconstruction. The default is 64.

    Returns
    -------
    proposed_ex_volt_mat : np.ndarray (M,4) with each row [current source, current sink, voltage+, voltage-)
        The newly proposed measurements, concatenated into a current pair and voltage pair matrix.
    repeated_ex : list int
        Returns any repeated current pairs to delete.

    """
    proposed_ex_volt_mat = []
    repeated_ex = np.array([]).astype(np.int32)
    for i in range(0, len(proposed_ex_lines)): # loop over the proposed current pairs
        num_repeats = 0
        # Generate the voltage measuring electrodes for this current driver pair
        proposed_voltage_pairs = measopt.findNextVoltagePair(proposed_ex_lines[i], save_filename, total_map, n_of_voltages, counter=0, npix=n_pix, cutoff=0.97)
        print("Proposed current pair:", proposed_ex_lines[i])
        #print("Proposed voltage pairs: size",len(proposed_voltage_pairs),"\n", proposed_voltage_pairs)
        for j in range(0, len(proposed_voltage_pairs)):
            line = np.hstack((proposed_ex_lines[i], proposed_voltage_pairs[j]))
            A = (line[0]==ex_volt_mat[:,0])
            B = (line[1]==ex_volt_mat[:,1])
            M = (line[2]==ex_volt_mat[:,2])
            N = (line[3]==ex_volt_mat[:,3])
            repeated_line_mask = A*B*M*N
            if (any(repeated_line_mask) == True): 
                num_repeats = num_repeats + 1
                continue # Skip any proposed voltages that repeat measurements already taken from a previous iteration
            elif len(proposed_ex_volt_mat)==0:
                proposed_ex_volt_mat = line
            else:
                proposed_ex_volt_mat = np.vstack((proposed_ex_volt_mat, line))
        if(num_repeats == len(proposed_voltage_pairs)):
            repeated_ex = np.append(repeated_ex, int(i))
    return proposed_ex_volt_mat, repeated_ex     

def getNextPrediction_partialforwardsolver(mesh_obj: None, volt_mat: None, ex_mat: np.ndarray, ind: None, voltages: np.ndarray, 
                      volt_mode:str='all', n_of_voltages: int=10, n_el: int=20, n_per_el: int=3, n_pix: int=64, pert: float=0.5, 
                                                                      p_influence: float=-10., p_rec: float=10.,p: float=0.5, lamb: float=0.1) -> np.ndarray:
    """
    Main adaptive ESA function. Obtains current and voltage pair proposals for measurements to be taken in some external code
    to this function. Performs one iteration of the adaptive ESA function, should be called multiple times via external code.
    
    Notes:
    This works with either real or simulated voltages. However, the true conductivity map can be obtained when using the pyEIT simulation method,
    which enables a more thorough and quantitative evaluation of the adaptive ESA performance. Note there are some issues with this, currently
    it does not properly account for repeated values. It deletes any repeated values, ensuring only unique measurements such that the two elements
    in the voltages array are never repeated measurements. However, it doesn't properly add new values to replace those deleted, meaning that the
    size of proposed_ex_volt_mat can vary in size from 1 to (current_pairs_to_propose * voltages_per_current_pair).
    Ensure that the 1D list of voltages matches the ex_volt_mat which is of form np.ndarray (M,4). Each row represents a single 4-terminal
    measurement -> [current source, current sink, voltage+, voltage-].

    Parameters
    ----------
    mesh_obj : None, mesh_obj class as implemented in pyEIT software
        Contains information on the mesh, such as electrode positions and nodes.
    volt_mat : int np.ndarray (M,2) with each row [current source, current sink]
        DESCRIPTION.
    ex_mat : int np.ndarray (M,2) with each row [current source, current sink]
        DESCRIPTION.
    ind : 1D array of int
        List of ints which can be used to map ex_mat onto correct dimensions for concatenation with volt_mat (ex_mat_extended = ex_mat[ind])
    voltages : 1D array of float
        List of voltage measurements from system. These are either real measurements or simulated, see demo below.
    volt_mode : str, optional
        The types of voltage pairs the ESA can propose. The default is 'all', can also be 'adj' or'opp'.
    n_of_voltages : int, optional
       Number of voltages to propose per current pair. The default is 10.
    n_el : int, optional
        Number of electrodes used. The default is 20.
    n_per_el : int, optional
        Number of nodes used in the mesh to represent the finite width of the electrodes for Complete Electrode Model. The default is 3.
    n_pix : int, optional
        Number of pixels for GREIT reconstruction resolution. The default is 64.
    pert : float, optional
        Hyperparameter of the adaptive ESA, determined during training. The default is 0.5.
    p_influence : float, optional
        Hyperparameter of the adaptive ESA, determined during training. The default is -10..
    p_rec : float, optional
        Hyperparameter of the adaptive ESA, determined during training. The default is 10..
    p : float, optional
        Noise covariance factor as used in GREIT reconstruction. The default is 0.5.
    lamb : float, optional
        Regularisation parameter as used in GREIT reconstruction. The default is 0.1.

    Returns
    -------
    proposed_ex_volt_mat : np.ndarray (M,4) with each row [current source, current sink, voltage+, voltage-]
        The newly proposed measurements, concatenated into a current pair and voltage pair matrix..
    ex_volt_mat : np.ndarray (M,4) with each row [current source, current sink, voltage+, voltage-]
        The previous current pair and voltage pair matrix with newly proposed measurements appended.
    ex_mat : int np.ndarray (M,2) with each row [current source, current sink]
        The previous current pairs with proposed current pairs appended.
    new_ind : int 1D array (M)
        The ind variable used to map the new ex_mat onto the new voltages.
    reconstruction : (n_pixels, n_pixels) array of float
        The GREIT reconstruction image.
    total_map : (n_pixels, n_pixels) array of float
        Contains the total map values for each pixel of the GREIT reconstruction, see adaptive ESA theory.

    """
    #ex_volt_mat = np.concatenate((ex_mat[ind], volt_mat), axis=1)
    # create an object with the meshing characteristics to initialise a Forward object
    el_pos = np.arange(n_el * n_per_el).astype(np.int16)
    if mesh_obj == None:
        mesh_obj = mesh(n_el)
    
    fwd = Forward_given(mesh_obj, el_pos, n_el)
    if (volt_mat is None):
        f, meas, new_ind = fwd.solve_eit(ex_mat=ex_mat)
        ex_volt_mat = meas
        #voltages = np.random.random(len(meas)) # Generate random voltage measurments for proof-of-concept
    #elif (volt_mat is not None):
    elif (volt_mat is not None) and (ind is not None):
        f, meas, new_ind = fwd.solve_eit(volt_mat=volt_mat, ex_mat=ex_mat, new_ind=ind)
        #ex_volt_mat = meas
        ex_volt_mat = np.concatenate((ex_mat[ind], volt_mat), axis=1)
       #ex_volt_mat=np.hstack((ex_mat[ind], volt_mat))
    else:
        print("Eh-ohhhh!")
    
    # now we can use the real voltage readings and the GREIT algorithm to reconstruct
    #volt_mat = ex_volt_meas[:, 2:]
    greit = train.greit.GREIT(mesh_obj, el_pos, f=f, ex_mat=(meas[:, :2]), step=None)
    greit.setup(p=p, lamb=lamb, n=n_pix)
    h_mat = greit.H
    reconstruction = greit.solve(voltages, f.v, normalize=False).reshape(n_pix, n_pix)
    # fix_electrodes_multiple is in meshing.py
    #_, el_coords_temp = train.fix_electrodes_multiple(centre=None, edgeX=0.1, edgeY=0.1, a=2, b=2, ppl=n_el, el_width=0.2, num_per_el=3)
    _, el_coords = train.fix_electrodes_multiple(centre=None, edgeX=0.1, edgeY=0.1, a=2, b=2, ppl=n_el, el_width=0.1, num_per_el=3, start_pos='mid')
    # find the distances between each existing electrode pair and the pixels lying on the line that connects them
    pixel_indices, voltage_all_possible = measopt.find_all_distances(reconstruction, h_mat, el_coords, n_el, cutoff=0.8, npix=n_pix)
    # call function get_total_map that generates the influence map, the gradient map and the log-reconstruction
    total_map, grad_mat, rec_log = np.abs(measopt.get_total_map(reconstruction, voltages, h_mat, pert=pert, p_influence=p_influence, p_rec=p_rec,npix=n_pix))
    # get the indices of the total map along the lines connecting each possible electrode pair
    total_maps_along_lines = total_map[None]*pixel_indices
    # find how close each connecting line passes to the boundary of an anomaly (where gradient supposed to be higher)
    proximity_to_boundary = np.sum(total_maps_along_lines, axis=(1, 2)) / np.sum(pixel_indices, axis=(1, 2))
    # rate the possible src-sink pairs by their proximity to existing anomalies
    proposed_ex_line = voltage_all_possible[np.argsort(proximity_to_boundary)[::-1]][:n_of_voltages]
    
    ex_line_start = 0
    ex_line_end = 2 # From 1 to desired number of current pairs
    while(True):
        new_ex_lines = proposed_ex_line[ex_line_start:ex_line_end]
        new_volt_mat, ind_temp = volt_matrix_generator(ex_mat=new_ex_lines, n_el=n_el, volt_pair_mode=volt_mode)
        # Regenerate Jacobian with new current pairs
        f, partial_meas, new_ind_2 = fwd.solve_eit(volt_mat=new_volt_mat, ex_mat=new_ex_lines, new_ind=ind_temp) 
        save_filename = 'relevant_jacobian_slice.h5'
        h = h5.File(save_filename, 'w')
        try:
            h.create_dataset('jac', data=f.jac)
            h.create_dataset('meas', data=partial_meas)
            h.create_dataset('p', data=mesh_obj['node'])
            h.create_dataset('t', data=mesh_obj['element'])
        except:
            TypeError('Error with saving files!')
        h.close()
        # Function call to obtain new voltage pairs appended to new current pairs
        proposed_ex_volt_mat, repeated_ex = propose_ex_volt_mat(new_ex_lines,
                            ex_volt_mat, save_filename, total_map, n_of_voltages=n_of_voltages, counter=0, n_pix=n_pix)
        #print("Proposed current and volt lines: size",len(proposed_ex_volt_mat),"\n", proposed_ex_volt_mat)
        if (ex_line_end >= len(proposed_ex_line)):
            print("No new unique voltages proposed and no remaining current pairs; breaking loop ...")
        elif ((ex_line_end+ex_line_end <= len(proposed_ex_line)) and (len(proposed_ex_volt_mat)==0)):
            print("No new unique voltages proposed; relooping over new current pairs ...")
            ex_line_start = ex_line_start + ex_line_end
            ex_line_end = ex_line_end + ex_line_end
        else:

            new_ex_lines = np.delete(new_ex_lines, repeated_ex, axis=0) 
            # if (len(new_ex_lines)==1):
            #     print(new_ex_lines)
            #     A = new_ex_lines[0]
            #     B = new_ex_lines[1]
            # else:
            #     A = new_ex_lines[0][0]
            #     B = new_ex_lines[0][1]
            #print("ex_lines:\n",new_ex_lines)
            A = new_ex_lines[0][0]
            B = new_ex_lines[0][1]
                
            if ((ex_mat[-1][0] == A) and (ex_mat[-1][1] == B) and (len(new_ex_lines)>1)):
                new_ex_lines = new_ex_lines[1:]
                ex_mat = np.vstack((ex_mat, new_ex_lines)) # Vertically stack new current pair onto excitation matrix
                print("New voltages added onto last current pair of previous ex_mat")
            elif ((ex_mat[-1][0] == A) and (ex_mat[-1][1] == B) and (len(new_ex_lines)==1)):
                ex_mat = ex_mat # no change, new voltage pairs added onto previous, last current pair
                print("Repeated current pair. No change to ex mat.")
            else:
                ex_mat = np.vstack((ex_mat, new_ex_lines))
            ex_volt_mat = np.vstack((ex_volt_mat, proposed_ex_volt_mat)) # ex_volt_mat == ex_volt_meas
            ex_mat, new_ind = ex_mat_ind_finder(ex_volt_mat)
            return proposed_ex_volt_mat, ex_volt_mat, ex_mat, new_ind, reconstruction, total_map

def adaptive_ESA_single_interation(mesh_obj: None, volt_mat: None, ex_mat: np.ndarray, ind: None, voltages: np.ndarray, 
                      volt_mode:str='all', n_of_voltages: int=10, n_el: int=32, n_per_el: int=3, n_pix: int=64, pert: float=0.5, 
                      p_influence: float=-10., p_rec: float=10., p: float=0.1, lamb: float=0.1, do_plot: bool=True) -> np.ndarray:
    """
    Calls the get_next_prediction function for a single iteration and can also plot the GREIT reconstruction.

    Parameters
    ----------
    mesh_obj : None, mesh_obj class as implemented in pyEIT software
        Contains information on the mesh, such as electrode positions and nodes.
    volt_mat : int np.ndarray (M,2) with each row [current source, current sink]
        DESCRIPTION.
    ex_mat : int np.ndarray (M,2) with each row [current source, current sink]
        DESCRIPTION.
    ind : 1D array of int
        List of ints which can be used to map ex_mat onto correct dimensions for concatenation with volt_mat (ex_mat_extended = ex_mat[ind])
    voltages : 1D array of float
        List of voltage measurements from system. These are either real measurements or simulated, see demo below.
    volt_mode : str, optional
        The types of voltage pairs the ESA can propose. The default is 'all', can also be 'adj' or'opp'.
    n_of_voltages : int, optional
       Number of voltages to propose per current pair. The default is 10.
    n_el : int, optional
        Number of electrodes used. The default is 20.
    n_per_el : int, optional
        Number of nodes used in the mesh to represent the finite width of the electrodes for Complete Electrode Model. The default is 3.
    n_pix : int, optional
        Number of pixels for GREIT reconstruction resolution. The default is 64.
    pert : float, optional
        Hyperparameter of the adaptive ESA, determined during training. The default is 0.5.
    p_influence : float, optional
        Hyperparameter of the adaptive ESA, determined during training. The default is -10..
    p_rec : float, optional
        Hyperparameter of the adaptive ESA, determined during training. The default is 10..
    p : float, optional
        Noise covariance factor as used in GREIT reconstruction. The default is 0.5.
    lamb : float, optional
        Regularisation parameter as used in GREIT reconstruction. The default is 0.1.
    do_plot: bool
        Activates the plotting feature or not, default is True, do the plot.

    Returns
    -------
    proposed_ex_volt_mat : np.ndarray (M,4) with each row [current source, current sink, voltage+, voltage-]
        The newly proposed measurements, concatenated into a current pair and voltage pair matrix.
    ex_volt_mat : np.ndarray (M,4) with each row [current source, current sink, voltage+, voltage-]
        The previous current pair and voltage pair matrix with newly proposed measurements appended.
    ex_mat : int np.ndarray (M,2) with each row [current source, current sink]
        The previous current pairs with proposed current pairs appended.
    ind : int 1D array (M)
        The ind variable used to map the new ex_mat onto the new voltages.
    reconstruction : (n_pixels, n_pixels) array of float
        The GREIT reconstruction image.
    total_map : (n_pixels, n_pixels) array of float
        Contains the total map values for each pixel of the GREIT reconstruction, see adaptive ESA theory.

    """
    proposed_ex_volt_lines, ex_volt_meas, ex_mat, ind, reconstruction, total_map = getNextPrediction_partialforwardsolver(mesh_obj=mesh_obj, 
                volt_mat=volt_mat, ex_mat=ex_mat, ind=ind, voltages=voltages,volt_mode=volt_mode, n_of_voltages=n_of_voltages, 
                n_el=n_el,p=p,lamb=lamb, n_pix=n_pix) 
    #volt_mat = ex_volt_meas[:, 2:]
    #print("\nSize of excitation matrix.")
    #print("No. of current pairs: ", len(ex_mat))
    #print("No. of voltage pairs: ", len(volt_mat))
    if (do_plot == True):
        plt.figure()
        im = plt.imshow(reconstruction, cmap=plt.cm.viridis, origin='lower', extent=[-1, 1, -1, 1])
        plt.title("GREIT Reconstruction\n(Current, Voltage) pairs: ("+str(len(ex_mat))+", "+str(len(volt_mat))+")")
        plt.colorbar(im)
        #plt.savefig(filepath+"ESA reconstruction "+str(i))
        plt.show()
    return proposed_ex_volt_lines, ex_volt_meas, ex_mat, ind, reconstruction, total_map


    
if __name__ == "__main__": ##################################################
    
    # you need to input several things: conducted measurements in shape (num_meas, 4), 
    # where first column is source, second sink and other two are voltage measuring electrodes
    # voltage readings in shape (num_meas) in the same order as conducted measurements
    n_el = 32
    n_per_el = 3
    n_pix = 64
    current_mode = 'opp'
    volt_mode = 'adj'
    ESA_volt_mode = 'all' # voltage pairs which the jacobian can try to calculate
    ex_mat_length = 10
    prediction_loops=10
    simulate_anomalies = True
    save_plots = False
    plot_set = 1
    demo_no = 1 # Demo_no=1 for a single iteration and plot. Demo_no=2 for many adaptive ESA loops and comparison to adj-adj method 
    
    if demo_no==2 and save_plots==True:
        filepath = os.getcwd() + "\\comparison plots\\set_"+str(plot_set)+"\\"
        try:
            os.mkdir(filepath)
        except:
            choice = input("set_"+str(plot_set)+" already exists, do you want to replace these plots? (y/n):")
            if (choice=="n"):
                exit(0)
            print("Overriding plots in folder set_"+str(plot_set))

    # Initialise current voltage excitation matrix to simulate some voltages measurements to kick start the ESA
    ex_volt_mat, ex_mat, volt_mat, ind = initialise_ex_volt_mat(current_mode=current_mode, 
                                                                volt_mode=volt_mode,n_el=n_el, ex_mat_length=ex_mat_length)
    mesh_obj = mesh(n_el=n_el,start_pos='mid') # Making an empty mesh
    el_pos = np.arange(n_el * n_per_el).astype(np.int16)
    #el_pos = np.arange(n_el).astype(np.int16)
    fwd = Forward_given(mesh_obj, el_pos, n_el)
    empty_mesh_f, empty_meas, empty_ind = fwd.solve_eit(volt_mat=volt_mat, new_ind=ind, ex_mat=ex_mat) # forward solve on the empty mesh
    
    # Either simulate data or modify code to read in some real data
    if simulate_anomalies is True:
        print("Simulating anomalies and voltage data")
        a = 2.0
        anomaly = train.generate_anoms(a, a)
        true = train.generate_examplary_output(a, int(n_pix), anomaly) # true conductivty map
        mesh_new = train.set_perm(mesh_obj, anomaly=anomaly, background=1) # New mesh with anomalies
        f_sim, dummy_meas, dummy_ind = fwd.solve_eit(volt_mat=volt_mat, new_ind=ind, ex_mat=ex_mat,
                                                                      perm=mesh_new['perm'].astype('f8'))
        greit = train.greit.GREIT(mesh_obj, el_pos, f=empty_mesh_f, ex_mat=ex_mat)
        greit.setup(p=0.2, lamb=0.01, n=n_pix)
        voltages = f_sim.v # assigning simulated voltages
        reconstruction_initial = greit.solve(voltages, empty_mesh_f.v).reshape(n_pix, n_pix)
        
        plt.figure()
        im1 = plt.imshow(true, cmap=plt.cm.viridis, origin='lower', extent=[-1, 1, -1, 1])
        plt.colorbar(im1)
        plt.title("True Image")
        if demo_no == 2 and save_plots==True: 
            plt.savefig(filepath+"True image")
        plt.show()
    reconstruction_plot(ex_volt_mat, voltages, mesh_obj, n_el=32, p=0.2, lamb=0.01, n_pix=64)     
        
    plt.figure()
    im2 = plt.imshow(reconstruction_initial, cmap=plt.cm.viridis, origin='lower', extent=[-1, 1, -1, 1])
    plt.title("Initial Reconstruction\n(Current, Voltage) pairs: ("+str(len(ex_mat))+", "+str(len(volt_mat))+")")
    plt.colorbar(im2)
    if demo_no == 2 and save_plots==True: 
        plt.savefig(filepath+"Initial Reconstruction")
    plt.show()


    if demo_no == 1: # SINGLE ITERATION DEMO
        proposed_ex_volt_lines, ex_volt_meas, ex_mat, ind, reconstruction, total_map = adaptive_ESA_single_interation(mesh_obj,
                                        volt_mat=volt_mat, ex_mat=ex_mat, ind=ind, voltages=voltages,volt_mode=ESA_volt_mode, n_of_voltages=30, n_el=32)
        volt_mat = ex_volt_meas[:, 2:] # New volt_mat called here since it is not output from func


    if demo_no == 2: # ADAPTIVE ESA LOOP AND ADJ-ADJ COMPARISON DEMO
        print("\nInitial size of excitation matrix.")
        print("No. of current pairs: ", len(ex_mat))
        print("No. of voltage pairs: ", len(volt_mat))
        
        for i in range(1,prediction_loops+1):
            print("\nLoop:",i," ------------------------")
            proposed_ex_volt_lines, ex_volt_meas, ex_mat, ind, reconstruction, total_map = getNextPrediction_partialforwardsolver(mesh_obj, 
                                            volt_mat=volt_mat, ex_mat=ex_mat, ind=ind, voltages=voltages, volt_mode=ESA_volt_mode, n_of_voltages=10, n_el=n_el) 
            volt_mat = ex_volt_meas[:, 2:]
            print("\nSize of excitation matrix.")
            print("No. of current pairs: ", len(ex_mat))
            print("No. of voltage pairs: ", len(volt_mat))
            if simulate_anomalies is True:
                f_sim, dummy_meas, dummy_ind = fwd.solve_eit(volt_mat=volt_mat, new_ind=ind, ex_mat=ex_mat,
                                                              perm=mesh_new['perm'].astype('f8'))
                voltages = f_sim.v # assigning simulated voltages
                
            if i in (1, prediction_loops//4, prediction_loops//2, 3*prediction_loops//4, prediction_loops):
                plt.figure()
                im = plt.imshow(reconstruction, cmap=plt.cm.viridis, origin='lower', extent=[-1, 1, -1, 1])
                plt.title("Reconstruction: "+str(i)+"\n(Current, Voltage) pairs: ("+str(len(ex_mat))+", "+str(len(volt_mat))+")")
                plt.colorbar(im)
                if demo_no == 2 and save_plots==True: 
                    plt.savefig(filepath+"ESA reconstruction "+str(i))
                plt.show()
                if i in (1, prediction_loops//4,3*prediction_loops//4):
                    plt.close()
                
        print("\nCreating Adj-Adj Comparison plots")           
        for i in (10,15,20,25,32):
            comp_ex_volt_mat, comp_ex_mat, comp_volt_mat, comp_ind = initialise_ex_volt_mat(current_mode='adj', volt_mode='adj', n_el=n_el, ex_mat_length=i)
            empty_mesh_f, empty_meas, empty_ind = fwd.solve_eit(ex_mat=comp_ex_mat)
            f_comp, comp_meas, comp_ind = fwd.solve_eit(volt_mat=empty_meas[:, 2:], new_ind=empty_ind, ex_mat=comp_ex_mat,perm=mesh_new['perm'].astype('f8'))
            greit = train.greit.GREIT(mesh_obj, el_pos, f=f_comp, ex_mat=comp_ex_mat, step=1)
            greit.setup(p=0.2, lamb=0.01, n=n_pix)
            h_mat = greit.H
            comp_reconstruction = greit.solve(f_comp.v, empty_mesh_f.v).reshape(n_pix, n_pix)
            
            plt.figure()
            im_comp = plt.imshow(comp_reconstruction, cmap=plt.cm.viridis, origin='lower', extent=[-1, 1, -1, 1])
            plt.title("Adj-Adj Reconstruction\n(Current, Voltage) pairs: ("+str(len(comp_ex_mat))+", "+str(len(empty_meas[:, 2:]))+")")
            plt.colorbar(im_comp)
            if demo_no == 2 and save_plots==True: 
                plt.savefig(filepath+"adj-adj reconstruction "+str(i))
            plt.show()
            if i in (10,20,25):
                plt.close()
    #plt.close('all')
    print("\nEnd of Script ------------------------")
    
"""
    Initialisation of ex_volt_mat
    ex_volt_mat format
    current pair 1 volt pair 1 
    current pair 1 volt pair 2
    current pair 1 volt pair 3
    current pair 1 volt pair 4
    current pair 1 volt pair 5
    current pair 2 volt pair 6
    current pair 2 volt pair 7
    current pair 2 volt pair 8
    current pair 2 volt pair 9
    ...
    ...
    current pair N volt pair M
    
    
    proposed_ex_volt_lines has one new current pair denoted A, and 10 corresponding new voltages
    ex_volt_meas = old_ex_volt_mat + Proposed_ex_volt_lines

    current pair 1 volt pair 1      Ex Mat: Current pair 1
    current pair 1 volt pair 2              Current pair 2
    current pair 1 volt pair 3              ....
    current pair 1 volt pair 4              Current pair N
    current pair 1 volt pair 5              Current pair A
    current pair 2 volt pair 6
    current pair 2 volt pair 7
    current pair 2 volt pair 8
    current pair 2 volt pair 9
    ...
    ...
    current pair N volt pair 
    Current pair A volt pair A1
    ...
    current pair A voltpair A10
    
    Now we want to use proposed_ex_volt_lines to get the new measurements. Pass proposed_ex_volt_lines (say size 10 for sake of it) into the
    Hardware measurement functions. This will gives us a new set of voltages (New_voltages -> 10 new values)
    Append these voltages onto the old list of voltages: voltages = np.append(voltages, New_voltages)
    
    To repeat we need volt_mat, ex_mat, ind and voltages
    
    Ex_mat is the array of all of the current pairs we have measured over
    volt_mat is the array of all of the voltage pairs we have measured over number_volt_pairs*number_current_pairs
    For every current pair (ex_line) we have say 5 voltage pairs

    Proposed_ex_volt_lines = an array of the new proposed measuremetns in Ex Mat + volt_mat format
    
    Ex_volt_meas = old
    expanded_ex_mat = ex_mat[ind]
    Pass proposed_ex_volt_lines into equipment and get corresponding voltages, in matchnig order
    # Append new voltage values onto old voltages voltages = np.append(voltages, New_meas_voltages)
""" 
