# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 15:59:57 2022
Author: Adam Coxson, MPhys Undergrad, The University of Manchester
Project: Automated Electrical Impedance Tomography for Graphene
Module: aesa_complexity_test.py
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
import os, time
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
    print_modes = False
    if (ex_pair_mode == 1 or ex_pair_mode == 'adj'):
        if print_modes == True: print("Adjacent current pairs mode")
        dist = 1
    elif (ex_pair_mode == n_el//2 or ex_pair_mode == 'opp'):
        if print_modes == True: print("Opposite current pairs mode")
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
    print_modes = False
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
        if print_modes == True: print("Adjacent voltage pairs mode")
        volt_mat, new_ex_mat, new_ind = measopt.voltMeterwStep(n_el, cp.asarray(ex_mat), step_arr=1, parser=None)
        ind = cp.asnumpy(new_ind)
        volt_mat = cp.asnumpy(volt_mat)
        
    elif (volt_pair_mode == n_el//2 or volt_pair_mode == 'opp'):
        if print_modes == True: print("Opposite voltage pairs mode")
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

def reconstruction_plot(ex_volt_mat, measured_voltages, mesh_obj=None, start_pos='left', n_el=32, n_per_el=3, p=0.2, lamb=0.01, n_pix=64):
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

        
def aesa_prediction(mesh_obj: None, volt_mat: None, ex_mat: np.ndarray, ind: None, voltages: np.ndarray, 
                      volt_mode:str='all', n_of_voltages: int=10, counter: np.ndarray=None, n_el: int=20, n_per_el: int=3, n_pix: int=64, pert: float=0.156208803, 
                                                                      p_influence: float=-14., p_rec: float=24., cutoff: int=0.69037305, p: float=0.5, lamb: float=0.1) -> np.ndarray:
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
    counter
    
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
        mesh(n_el=n_el, num_per_el=n_per_el, start_pos='left', ref_perm=0.03352)
    
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
    _, el_coords = train.fix_electrodes_multiple(centre=None, edgeX=0.08, edgeY=0.08, a=2, b=2, ppl=n_el, el_width=0.04, num_per_el=n_per_el, start_pos='left')
    # find the distances between each existing electrode pair and the pixels lying on the line that connects them
    pixel_indices, voltage_all_possible = measopt.find_all_distances(reconstruction, h_mat, el_coords, n_el, cutoff=cutoff, npix=n_pix)
    # call function get_total_map that generates the influence map, the gradient map and the log-reconstruction
    total_map, grad_mat, rec_log = np.abs(measopt.get_total_map(reconstruction, voltages, h_mat, pert=pert, p_influence=p_influence, p_rec=p_rec,npix=n_pix))
    # get the indices of the total map along the lines connecting each possible electrode pair
    total_maps_along_lines = total_map[None]*pixel_indices
    # find how close each connecting line passes to the boundary of an anomaly (where gradient supposed to be higher)
    proximity_to_boundary = np.sum(total_maps_along_lines, axis=(1, 2)) / np.sum(pixel_indices, axis=(1, 2))
    # rate the possible src-sink pairs by their proximity to existing anomalies
    proposed_ex_line = voltage_all_possible[np.argsort(proximity_to_boundary)[::-1]][:5] # 10 proposed current pairs
    
    ex_line_start = 0
    ex_line_end = 10 # From 1 to desired number of current pairs
    ex_mat_all = train.generateExMat(ne=n_el)
    if counter is None:
        counter = np.zeros(len(ex_mat_all))
    new_ex_lines = proposed_ex_line[ex_line_start:ex_line_end]
    new_volt_mat, ind_temp = volt_matrix_generator(ex_mat=new_ex_lines, n_el=n_el, volt_pair_mode=volt_mode)
    # Regenerate Jacobian with new current pairs
    f, partial_meas, new_ind_2 = fwd.solve_eit(volt_mat=new_volt_mat, ex_mat=new_ex_lines, new_ind=ind_temp) 
    save_filename = 'aesa_jacobian_slice.h5'
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
    # proposed_ex_volt_mat, repeated_ex = propose_ex_volt_mat(new_ex_lines,
    #                     ex_volt_mat, save_filename, total_map, n_of_voltages=n_of_voltages, counter=0, n_pix=n_pix)
    j=0
    while True:
        loc = (np.sum((ex_mat_all == new_ex_lines[None, j]), axis=1) == 2).astype(bool)
        #if counter[loc] == 0 or counter[loc] < (n_el - 2) * (n_el - 3)//(10*n_of_voltages):
        if counter[loc] == 0 or counter[loc] < 2:
            counter[loc] += 1
            new_ex_lines = new_ex_lines[j]
            proposed_voltage_pairs = measopt.findNextVoltagePair(new_ex_lines, save_filename,
                                                        total_map, n_of_voltages, counter[loc], npix=n_pix, cutoff=cutoff)
            #print("new voltages:\n",proposed_voltage_pairs)
            break
        else:
            j += 1
    proposed_ex_volt_mat = [0]*len(proposed_voltage_pairs)
    for j in range(0, len(proposed_voltage_pairs)): proposed_ex_volt_mat[j]=np.hstack((new_ex_lines, proposed_voltage_pairs[j]))
    proposed_ex_volt_mat=np.array(proposed_ex_volt_mat)

    ex_mat = np.vstack((ex_mat, new_ex_lines))
    ex_volt_mat = np.vstack((ex_volt_mat, proposed_ex_volt_mat)) # ex_volt_mat == ex_volt_meas
    ex_mat, new_ind = ex_mat_ind_finder(ex_volt_mat)
    return proposed_ex_volt_mat, ex_volt_mat, ex_mat, new_ind, reconstruction, total_map, counter

if __name__ == "__main__": ##################################################
    
    # you need to input several things: conducted measurements in shape (num_meas, 4), 
    # where first column is source, second sink and other two are voltage measuring electrodes
    # voltage readings in shape (num_meas) in the same order as conducted measurements
    a = 2.0
    n_anomalies = 2
    n_per_el = 3
    n_pix = 64
    current_mode = 'opp'
    volt_mode = 'adj'
    ESA_volt_mode = 'all' # voltage pairs which the jacobian can try to calculate
    ex_mat_length = None
    prediction_loops=5
    
    
    n_el_list = [24, 32, 40, 48, 56, 64]
    n_el_list = [24, 32]
    t_list = np.zeros(len(n_el_list))
    
    for k in range(len(n_el_list)):
        n_el=n_el_list[k]
        # Initialise current voltage excitation matrix to simulate some voltages measurements to kick start the ESA
        ex_volt_mat, ex_mat, volt_mat, ind = initialise_ex_volt_mat(current_mode=current_mode, volt_mode=volt_mode,n_el=n_el, 
                                                                    ex_mat_length=ex_mat_length, ex_spaced=False)
        n_current_pairs = len(ex_mat)
        n_volt_per_current = len(volt_mat)/len(ex_mat)
        n_volt_pairs = len(volt_mat)
        n_tot = n_volt_pairs//2
        volts_per_set = n_tot//10
        
        
        mesh_params = [0.081,2500,9.55,3.14]
        #mesh_params = [0.054, 2500, 10., 10.]
    
        aesa = [0.916,-20,0.446,20] # Pretty good :) but only after hours of trialling =/ 
    
        p = 0.2
        lamb=0.01
        mesh_obj = mesh(n_el=n_el, num_per_el=n_per_el, el_width=0.04, edge=0.08, mesh_params=mesh_params, start_pos='left') # Making an empty mesh
        el_pos = np.arange(n_el * n_per_el).astype(np.int16)
        fwd = Forward_given(mesh_obj, el_pos, n_el)
        empty_mesh_f, empty_meas, empty_ind = fwd.solve_eit(volt_mat=volt_mat, new_ind=ind, ex_mat=ex_mat)
        
        t_sum=0
        print("Simulating anomalies and voltage data")
        anomaly_list = [0]*n_anomalies
        for i in range(n_anomalies):
            anomaly_list[i] = train.generate_anoms(a, a)
        
        for anom in range(0, n_anomalies):
            anomaly = anomaly_list[anom]
            
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
            plt.show()
            reconstruction_plot(ex_volt_mat, voltages, mesh_obj, n_el=n_el, p=0.2, lamb=0.01, n_pix=n_pix)     
                
            plt.figure()
            im2 = plt.imshow(reconstruction_initial, cmap=plt.cm.viridis, origin='lower', extent=[-1, 1, -1, 1])
            plt.title("Initial Reconstruction\n(Current, Voltage) pairs: ("+str(len(ex_mat))+", "+str(len(volt_mat))+")")
            plt.colorbar(im2)
            plt.show()
            
            
            print("\nInitial size of excitation matrix.")
            print("No. of current pairs: ", len(ex_mat))
            print("No. of voltage pairs: ", len(volt_mat))
            counter=None
            for i in range(1,prediction_loops+1):
                print("\nLoop:",i," ------------------------")
                t1 = time.time()
                proposed_ex_volt_lines, ex_volt_meas, ex_mat, ind, reconstruction, total_map, counter = aesa_prediction(mesh_obj, 
                                                volt_mat=volt_mat, ex_mat=ex_mat, ind=ind, voltages=voltages, volt_mode=ESA_volt_mode, n_of_voltages=volts_per_set,
                                                n_el=n_el,counter=counter, n_per_el=n_per_el,p=p,lamb=lamb,n_pix=n_pix,
                                            cutoff=aesa[0],p_influence=aesa[1],pert=aesa[2],p_rec=aesa[3]) 
                t2 = time.time()
                t_sum = t_sum + t2-t1
                volt_mat = ex_volt_meas[:, 2:]
                print("\nSize of excitation matrix.")
                print("No. of current pairs: ", len(ex_mat))
                print("No. of voltage pairs: ", len(volt_mat))
                f_sim, dummy_meas, dummy_ind = fwd.solve_eit(volt_mat=volt_mat, new_ind=ind, ex_mat=ex_mat,
                                                              perm=mesh_new['perm'].astype('f8'))
                voltages = f_sim.v # assigning simulated voltages
                    
                # if i in (1, prediction_loops//4, prediction_loops//2, 3*prediction_loops//4, prediction_loops):
                #     plt.figure()
                #     im = plt.imshow(reconstruction, cmap=plt.cm.viridis, origin='lower', extent=[-1, 1, -1, 1])
                #     plt.title("Reconstruction: "+str(i)+"\n(Current, Voltage) pairs: ("+str(len(ex_mat))+", "+str(len(volt_mat))+")")
                #     plt.colorbar(im)
                #     plt.show()
                #     if i in (1, prediction_loops//4,3*prediction_loops//4):
                #         plt.close()
            reconstruction_plot(ex_volt_meas, voltages, mesh_obj, n_el=n_el, n_per_el=n_per_el, p=0.2, lamb=0.01, n_pix=n_pix) 
        t_list[k]=t_sum