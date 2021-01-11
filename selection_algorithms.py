"""
Created Oct 2020
Authors: Frederik Brookebarnes and Adam Coxson, MPhys Undergrads, The University of Manchester
Project: Automated Electrical Impedance Tomography of Graphene
Module:  selection algorithms.py
Dependancies: get_next_prediction.py (from adaptive ESA), meshing.py (from modified pyEIT)

This works with component_control.py which takes four-terminal current and
voltage measurements for electrical impedance tomography setups. component_control is integrated
with the pyEIT package: https://github.com/liubenyuan/pyEIT

This module contains functions for different electrode selection algorithms which are then
called in component_control.py. These are used to provide lists of current pairs and voltage pairs
in the required format of (I+,I-,V+,V-) for the listed measurements to be taken. This also 
provides functionality for integration of the adaptive ESA.

"""
import numpy as np
import numpy.random as random
from get_next_prediction import adaptive_ESA_single_interation as adaptive_ESA
from get_next_prediction import initialise_ex_volt_mat
from meshing import mesh


def eit_scan_lines(ne=16, dist=1):
    """
    generate scan matrix
    Parameters
    ----------
    ne: int
        number of electrodes
    dist: int
        distance between A and B (default=1)
    Returns
    -------
    ex_mat: NDArray
        stimulation matrix
    Notes
    -----
    in the scan of EIT (or stimulation matrix), we use 4-electrodes
    mode, where A, B are used as positive and negative stimulation
    electrodes and M, N are used as voltage measurements
    1 (A) for positive current injection,
    -1 (B) for negative current sink
    dist is the distance (number of electrodes) of A to B
    in 'adjacent' mode, dist=1, in 'apposition' mode, dist=ne/2
    Examples
    --------
    # let the number of electrodes, ne=16
    if mode=='neighbore':
        ex_mat = eit_scan_lines()
    elif mode=='apposition':
        ex_mat = eit_scan_lines(dist=8)
    WARNING
    -------
    ex_mat is a local index, where it is ranged from 0...15, within the range
    of the number of electrodes. In FEM applications, you should convert ex_mat
    to global index using the (global) el_pos parameters.
    """
    ex = np.array([[i, np.mod(i + dist, ne)] for i in range(ne)])

    return ex

def voltage_meter(ex_line, n_el=16, step=1, parser=None):
    """
    extract subtract_row-voltage measurements on boundary electrodes.
    we direct operate on measurements or Jacobian on electrodes,
    so, we can use LOCAL index in this module, do not require el_pos.
    Notes
    -----
    ABMN Model.
    A: current driving electrode,
    B: current sink,
    M, N: boundary electrodes, where v_diff = v_n - v_m.
    'no_meas_current': (EIDORS3D)
    mesurements on current carrying electrodes are discarded.
    Parameters
    ----------
    ex_line: NDArray
        2x1 array, [positive electrode, negative electrode].
    n_el: int
        number of total electrodes.
    step: int
        measurement method (two adjacent electrodes are used for measuring).
    parser: str
        if parser is 'fmmu', or 'rotate_meas' then data are trimmed,
        boundary voltage measurements are re-indexed and rotated,
        start from the positive stimulus electrodestart index 'A'.
        if parser is 'std', or 'no_rotate_meas' then data are trimmed,
        the start index (i) of boundary voltage measurements is always 0.
    Returns
    -------
    v: NDArray
        (N-1)*2 arrays of subtract_row pairs
    """
    # local node
    drv_a = ex_line[0]
    drv_b = ex_line[1]
    i0 = drv_a if parser in ("fmmu", "rotate_meas") else 0
    # build differential pairs
    v = []
    for a in range(i0, i0 + n_el):
        m = a % n_el
        n = (m + step) % n_el
        # if any of the electrodes is the stimulation electrodes
        if not (m == drv_a or m == drv_b or n == drv_a or n == drv_b): 
            # the order of m, n matters
            v.append([n, m])

    diff_pairs = np.array(v)
    return diff_pairs

def Standard(no_electrodes, step=1, parser=None, dist=1):
    '''
    Inputs
    ------
    no_electrodes: int
        specifies number of electrodes
    step: int
        specifies the pattern to take measurements at (see voltage_meter), 
        step=1 for adj-adj, two adjacent electrodes are used for measuring
    parser: str
        parser: str
        if parser is 'fmmu', or 'rotate_meas' then data are trimmed,
        boundary voltage measurements are re-indexed and rotated,
        start from the positive stimulus electrodestart index 'A'.
        if parser is 'std', or 'no_rotate_meas' then data are trimmed,
        the start index (i) of boundary voltage measurements is always 0.
    
    Output
    ------
    electrodes: NDarray
        no_electrodes array*4 array of electrode numbers in the order ABMN, (sin+, sin-, v+, v-)
    '''
    #print(parser)
    scan_lines = eit_scan_lines(no_electrodes, dist=dist) # Generate current pairs separated by step=dist
    electrodes = []
    for i in range(0, len(scan_lines)):
        measurements = voltage_meter(scan_lines[i], n_el=no_electrodes, step=step, parser=parser) # add voltage excitation pairs for current pairs
        for j in range(0, len(measurements)):
            electrodes.append(np.concatenate((scan_lines[i], measurements[j])))
    electrodes = np.array(electrodes)
    return electrodes 


def GetNextElectrodes(algorithm='Standard', no_electrodes=32, all_measurement_electrodes=None, measurement=0, **algorithm_parameters):
    '''
    Returns electrode connections (eg sin+:2, sin-:1, v+: 18, v-:17) given the algorithm used 
    and required information eg measurement no. or previous measurement, in order of (sin+, sin-, v+, v-).
    If a list of electrodes are already given, it simply returns the nth element in that array. 
    
    Inputs
    ------
    algorithm: str
        Electrode selection algorithm to be used
    no_electrodes: int
        Number of electrodes
    all_measurement_electrodes: NDarray
        A 4*N array of electrode positions for all measurements. Allows user to pre-generate desired electrode positions instead of using algorithm.
        Helps to speed up when using Standard algorithm.
    measurement: 0
        Specifies which measurement the EIT process is on.
    algorithm_parameters: **kwargs
        Allows user to pass relevant parameters of desired algorithm
    
    Output
    ------
    next_electrodes: NDarray
        4*N array of electrode positions to take next measurement from
    not_last_measurement: bool
        If no more measurements should be taken not_last_measurement=False, otherwise if more measurements should be taken not_last_measurement=True
    '''
    not_last_measurement = True
    next_electrodes = np.zeros(4)

    #print(algorithm_parameters)
    if algorithm == 'Standard':
        try:
            if all_measurement_electrodes == None:
                all_measurement_electrodes = Standard(no_electrodes, **algorithm_parameters)
        except:
            #print("list given")
            pass
        if measurement >= len(all_measurement_electrodes):
            not_last_measurement = False  
        if not_last_measurement == True:
            next_electrodes = all_measurement_electrodes[measurement]
            
    if algorithm == 'Random':
        rng = random.default_rng()
        next_electrodes = rng.choice(no_electrodes-1, size=4, replace=False)
        
    if algorithm == 'ESA':
        n_el = no_electrodes
        if measurement == 0:
            # QWARGS: initialise(current_mode:str='adj', volt_mode:str='adj',n_el:int=32, ex_mat_length:int=10, ex_spaced:bool=False)
            #ex_volt_mat, ex_mat, volt_mat, ind = initialise_ex_volt_mat(n_el=n_el,**algorithm_parameters)
            ex_volt_mat, ex_mat, volt_mat, ind = initialise_ex_volt_mat(current_mode='opp', volt_mode='adj',n_el=32, ex_mat_length=32, ex_spaced=False)
            mesh_obj = mesh(n_el=n_el,start_pos='mid') # mid for our resistor grid since electrode 0 is at the centre of a side
            next_electrodes = ex_volt_mat
            ex_volt_index_data = (ex_volt_mat, ex_mat, volt_mat, ind, mesh_obj)
        else:
            #GetNextElectrodes(*algorithm_parameters)
            #voltages = np.random.random(len(ex_volt_mat)) # random voltages to ensure script working
            #QWARGS: adaptive_ESA(mesh_obj,volt_mat, ex_mat, ind, voltages, num_returned, n_el=32, do_plot=True):
            proposed_ex_volt_lines, ex_volt_meas, ex_mat, ind, reconstruction, total_map = adaptive_ESA(n_el=n_el, num_returned=10, **algorithm_parameters)
            volt_mat = ex_volt_meas[:, 2:]
            next_electrodes = proposed_ex_volt_lines
            ex_volt_index_data = (ex_volt_meas, ex_mat, volt_mat, ind, mesh_obj)
            return next_electrodes, not_last_measurement, ex_volt_index_data

    return next_electrodes, not_last_measurement


def GetNextElectrodesESA(voltages, ESA_params, ex_volt_index_data=None, n_el=32, measurement=0, max_measurement=3):
    """
    Returns electrode connections (eg sin+:2, sin-:1, v+: 18, v-:17) given the algorithm used 
    and required information eg measurement no. or previous measurement, in order of (sin+, sin-, v+, v-).
    If a list of electrodes are already given, it simply returns the nth element in that array. 
    
    Due to code incompatibily, this second version of the above function was made as a stop gap solution. This 
    interfaces with getNextPrediction_partialforwardsolver() and component_control.py to enable measurements and 
    operation of the adaptive ESA. 
    
    To Do:
        Organise the selection algorithms to better account for the different operations.

    Parameters
    ----------
    voltages: 1D array of floats
        list of measured voltages corresponding to format of ex_volt_mat.
    ESA_params : TYPE
        DESCRIPTION.
    ex_volt_index_data : tuple of (ex_volt_mat, ex_mat, volt_mat, ind, mesh_obj) 
       This contains the np.ndarray and class object variables required for reiteration of the adaptive ESA, wrapped in a tuple for ease.
       The default is None, in which case, it creates it and returns for future iteration loops of the adaptive ESA.
    n_el : int, optional
        Number of electrodes. The default is 32.
    measurement : int, optional
        To track current measurement iteration. The default is 0.
    max_measurement : int, optional
        Upper bound of measurements to be taken before cutting loop short. The default is 3.

    Returns
    -------
    proposed_ex_volt_lines : np.ndarray (M,4) with each row [current source, current sink, voltage+, voltage-]
        The newly proposed measurements, concatenated into a current pair and voltage pair matrix..
    ex_volt_index_data : tuple of (ex_volt_mat, ex_mat, volt_mat, ind, mesh_obj) 
       This contains the np.ndarray and class object variables required for reiteration of the ESA, wrapped in a tuple for ease.
    """

    #print("ex_volt_index_data: ", ex_volt_index_data)
    #ESA_params = (current_mode, volt_mode, current_pairs_to_return, voltages_to_return, n_pix, pert, p_influence, p_rec)
    current_mode            = ESA_params[0]
    volt_mode               = ESA_params[1]
    ESA_volt_mode           = ESA_params[2]
    current_pairs_to_return = ESA_params[3] # Currently hard coded to 3 current pairs per loop within get_next_prediction
    voltages_to_return      = ESA_params[4]
    n_pix                   = ESA_params[5] # Currently redundant as defaults are used
    pert                    = ESA_params[6] # Currently redundant as defaults are used
    p_influence             = ESA_params[7] # Currently redundant as defaults are used
    p_rec                   = ESA_params[8] # Currently redundant as defaults are used
    ex_mat_length           = ESA_params[9]
    spaced_ex_mat           = ESA_params[10]
    p                       = ESA_params[11]
    lamb                    = ESA_params[12]
    
    if (measurement == 0 or ex_volt_index_data == None):
        print("No measurements inputted, initialising ex_volt_mat using current and volt mode")
        ex_volt_mat, ex_mat, volt_mat, ind = initialise_ex_volt_mat(current_mode=current_mode, volt_mode=volt_mode,n_el=n_el,
                                                                    ex_mat_length=ex_mat_length, ex_spaced=spaced_ex_mat)
        mesh_obj = mesh(n_el=n_el,start_pos='mid') # mid for our resistor grid since electrode 0 is at the centre of a side
        ex_volt_index_data = (ex_volt_mat, ex_mat, volt_mat, ind, mesh_obj)
        proposed_ex_volt_lines = ex_volt_mat
    else:
        ex_mat = ex_volt_index_data[1]    
        volt_mat = ex_volt_index_data[2]
        ind = ex_volt_index_data[3]
        mesh_obj = ex_volt_index_data[4]
        #print("ex_mat", ex_mat)
        #print("volt_mat", volt_mat)
        #print("ind", ind)
        #print("mesh_obj", mesh_obj)
        #print("m:",measurement," max:",max_measurement-1)
        # if (measurement in [1,2,3,4])
        if ((measurement == 1) or (measurement == max_measurement-2)):
            plot=True
        else:
            plot=False
        plot=True
        print("Measurement", measurement,". Do plot? ", plot)
        proposed_ex_volt_lines, ex_volt_meas, ex_mat, ind, sigma, total = adaptive_ESA(n_el=n_el, n_of_voltages=voltages_to_return, volt_mode=ESA_volt_mode,
                 mesh_obj=mesh_obj,volt_mat=volt_mat, ex_mat=ex_mat, ind=ind, voltages=voltages, p=p,n_pix=n_pix, lamb=lamb, do_plot=plot)
        volt_mat = ex_volt_meas[:, 2:]
        ex_volt_index_data = (ex_volt_meas, ex_mat, volt_mat, ind, mesh_obj)

    return proposed_ex_volt_lines, ex_volt_index_data

'''
next_electrodes, keep_measuring, ex_volt_index = GetNextElectrodesESA(voltages=None, max_measurement=2, no_electrodes=32, all_measurement_electrodes=None, measurement=0,ex_volt_index_data=None)

print(next_electrodes, keep_measuring, ex_volt_index)
'''


'''
def RunEIT(algorithm='Standard', no_electrodes=32, max_measurements=None, measurement_electrodes = None, **algorithm_parameters):

    ClearSwitches()


    #standard_measurement_electrodes = Standard(no_electrodes=6, step=1,parser='fmmu')

    #print(standard_measurement_electrodes)


    keep_measuring = True

    if max_measurements == None:
        max_measurements = 10000

    v_difference = []

    while keep_measuring == True:
        for i in range(0,max_measurements):

            next_electrodes, keep_measuring = GetNextElectrodes(algorithm=algorithm, no_electrodes=no_electrodes, measurement=i)
            print("measurement "+str(i)+", next electrode "+str(next_electrodes)+"keep measuring:"+str(keep_measuring))
            if keep_measuring == False:
                break
            print(next_electrodes)
            ClearSwitches()
            for i in next_electrodes:
                FlickSwitch(on, MapSwitches(electrode=next_electrodes[i], lockin_connection=i))
            r, theta, samp, fint = GetMeasurement(param_set=False)
            v_difference.append(r)
        v_difference = np.array(v_diff)

    return return v_difference
'''

#RunEIT(no_electrodes=6, max_measurements=1000)

# def GetNextElectrodesESA_old(voltages, max_measurement, no_electrodes=32, all_measurement_electrodes=None, measurement=0,ex_volt_index_data=None):
#     '''
#     Returns electrode connections (eg sin+:2, sin-:1, v+: 18, v-:17) given the algorithm used 
#     and required information eg measurement no. or previous measurement, in order of (sin+, sin-, v+, v-).
#     If a list of electrodes are already given, it simply returns the nth element in that array. 
    
#     Inputs
#     ------
#     algorithm: str
#         Electrode selection algorithm to be used
#     no_electrodes: int
#         Number of electrodes
#     all_measurement_electrodes: NDarray
#         A 4*N array of electrode positions for all measurements. Allows user to pre-generate desired electrode positions instead of using algorithm.
#         Helps to speed up when using Standard algorithm.
#     measurement: 0
#         Specifies which measurement the EIT process is on.
#     algorithm_parameters: **kwargs
#         Allows user to pass relevant parameters of desired algorithm
    
#     Output
#     ------
#     next_electrodes: NDarray
#         4*N array of electrode positions to take next measurement from
#     not_last_measurement: bool
#         If no more measurements should be taken not_last_measurement=False, otherwise if more measurements should be taken not_last_measurement=True
#     '''

#     algorithm = 'ESA'
#     not_last_measurement = True
#     next_electrodes = np.zeros(4)

#     #print(algorithm_parameters)
#     if algorithm == 'Standard':
#         if all_measurement_electrodes == None:
#             all_measurement_electrodes = Standard(no_electrodes, **algorithm_parameters)
#         if measurement >= len(all_measurement_electrodes):
#             not_last_measurement = False  
#         if not_last_measurement == True:
#             next_electrodes = all_measurement_electrodes[measurement]
            
#     if algorithm == 'Random':
#         rng = random.default_rng()
#         next_electrodes = rng.choice(no_electrodes-1, size=4, replace=False)
        
#     if algorithm == 'ESA':
#         n_el = no_electrodes
#         print("ex_volt_index_data", ex_volt_index_data)
#         if measurement == 0:
#             # QWARGS: initialise(current_mode:str='adj', volt_mode:str='adj',n_el:int=32, ex_mat_length:int=10, ex_spaced:bool=False)
#             #ex_volt_mat, ex_mat, volt_mat, ind = initialise_ex_volt_mat(n_el=n_el,**algorithm_parameters)
#             ex_volt_mat, ex_mat, volt_mat, ind = initialise_ex_volt_mat(current_mode='adj', volt_mode='opp',n_el=32, ex_mat_length=None, ex_spaced=False)
#             mesh_obj = mesh(n_el)
#             next_electrodes = ex_volt_mat
#             ex_volt_index_data = (ex_volt_mat, ex_mat, volt_mat, ind, mesh_obj)
#         else:
#             ex_mat = ex_volt_index_data[1]    # named for ESA KWARGs
#             volt_mat = ex_volt_index_data[2]
#             ind = ex_volt_index_data[3]
#             mesh_obj = ex_volt_index_data[4]
#             print("ex_mat", ex_mat)
#             print("volt_mat", volt_mat)
#             print("ind", ind)
#             #print("mesh_obj", mesh_obj)
#             print("m:",measurement," max:",max_measurement-1)
#             if measurement == max_measurement-1:
#                 plot=True
#             else:
#                 plot=False
#             print("plot:",plot)
#             #QWARGS: adaptive_ESA(mesh_obj,volt_mat, ex_mat, ind, voltages, num_returned, n_el=32, do_plot=True):
#             proposed_ex_volt_lines, ex_volt_meas, ex_mat, ind, sigma, total = adaptive_ESA(n_el=n_el, num_returned=10,
#                      mesh_obj=mesh_obj,volt_mat=volt_mat, ex_mat=ex_mat, ind=ind, voltages=voltages, do_plot=plot)
#             volt_mat = ex_volt_meas[:, 2:]
#             next_electrodes = proposed_ex_volt_lines
#             ex_volt_index_data = (ex_volt_meas, ex_mat, volt_mat, ind, mesh_obj)

#     return next_electrodes, not_last_measurement, ex_volt_index_data