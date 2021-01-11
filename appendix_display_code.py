# -*- coding: utf-8 -*-
"""
Author: Adam Coxson, MPhys Undergrad, The University of Manchester
Project: Machine learning enabled Electrical Impedance Tomography for 2D materials
This file is meant to display the relevant code myself and my lab partner Frederik Brooke Barnes created during the 1st Semester
MPhys project. Below is a breakdown of the functions and their respective .py files. Note, only the main and most used functions 
are shown. For the full code see my github: https://github.com/AdamCoxson
No functions from the modified pyEIT software are shown. For those files, see Ivo Mihovs and Vasil Avramovs MPhys papers.

get_next_prediction.py: (pyEIT based)
    This was the interface module between the pyVisa lock-in and switch box control scripts and the rest of the pyEIT software.
    It contained functions which formatted the desired measurements and read in the subsequent voltage data for analysis.
    - initialise_ex_volt_mat
    - proposed_ex_volt_mat
    - getNextPrediction_partialforwardsolver
    - adaptive_ESA
    
selection_algorithms.py: (pyVisa based)
    A module called within component_control, contains methods for the different selection algorithms for application to the
    measurement hardware.
    - Standard
    - GetNextElectrodes
    - GetNextElectrodesESA
    
component_control.py: (pyVisa based)
    The main pyVisa module which contain functions for controlling the switchbox and lock-in measurement operation.
    - SetMeasurementParameters
    - GetMeasurement
    - FlickSwitch
    - MapSwitches
    - ClearSwitches
    - wait_test
    - RunEIT

##############################################################################
 The following functions belong to the get_next_prediction.py module.
    This was the interface module between the pyVisa lock-in and switch box control scripts and the rest of the pyEIT software.
    It contained functions which formatted the desired measurements and read in the subsequent voltage data for analysis.
    - initialise_ex_volt_mat
    - proposed_ex_volt_mat
    - getNextPrediction_partialforwardsolver
    - adaptive_ESA
##############################################################################
""" 

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

def propose_ex_volt_mat(proposed_ex_lines, ex_volt_mat, save_filename, total_map, n_of_voltages=10, counter=0, n_pix=64):
    """
    This finds and proposes new voltage pairs for the given current pairs. For use in adaptive ESA. 
    
    Notes:
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
    proposed_ex_lines : int np.ndarray (M,2) with each row [current source, current sink]
        The current pair matrix of new proposed current pairs obtained from analysing the total map.
    ex_volt_mat : np.ndarray (M,4) with each row [current source, current sink, voltage+, voltage-]
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
                                                                      p_influence: float=-10., p_rec: float=10.,
                                                                      p: float=0.5, lamb: float=0.1) -> np.ndarray:
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
        The newly proposed measurements, concatenated into a current pair and voltage pair matrix.
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
    # Implement mesh passed in via Args or create a meshing object to initialise a Forward object
    el_pos = np.arange(n_el * n_per_el).astype(np.int16)
    if mesh_obj == None:
        mesh_obj = mesh(n_el)
    fwd = Forward_given(mesh_obj, el_pos, n_el)
    if (volt_mat is None):
        f, meas, new_ind = fwd.solve_eit(ex_mat=ex_mat)
        ex_volt_mat = meas
    elif (volt_mat is not None) and (ind is not None):
        f, meas, new_ind = fwd.solve_eit(volt_mat=volt_mat, ex_mat=ex_mat, new_ind=ind)
        ex_volt_mat = np.concatenate((ex_mat[ind], volt_mat), axis=1)
    else:
        print("Something went wrong in the forward solver")
    
    #  Use the real voltage readings and the GREIT algorithm to reconstruct
    greit = train.greit.GREIT(mesh_obj, el_pos, f=f, ex_mat=(meas[:, :2]), step=None)
    greit.setup(p=p, lamb=lamb, n=n_pix)
    h_mat = greit.H
    reconstruction = greit.solve(voltages, f.v, normalize=False).reshape(n_pix, n_pix)
    _, el_coords = train.fix_electrodes_multiple(centre=None, edgeX=0.1, edgeY=0.1, a=2, b=2, ppl=n_el, el_width=0.1, num_per_el=3, start_pos='mid')
    
    # Crrent pair electrode selection, find distances between each existing electrode pair and the pixels lying on the line that connects them
    pixel_indices, voltage_all_possible = measopt.find_all_distances(reconstruction, h_mat, el_coords, n_el, cutoff=0.8, npix=n_pix)
    
    # Call function get_total_map that generates the influence map, the gradient map and the log-reconstruction
    total_map, grad_mat, rec_log = np.abs(measopt.get_total_map(reconstruction, voltages, h_mat, pert=pert, p_influence=p_influence,
                                                                p_rec=p_rec,npix=n_pix))
    
    # Get the indices of the total map along the lines connecting each possible electrode pair
    total_maps_along_lines = total_map[None]*pixel_indices
    
    # Find how close each connecting line passes to the boundary of an anomaly (where gradient supposed to be higher)
    proximity_to_boundary = np.sum(total_maps_along_lines, axis=(1, 2)) / np.sum(pixel_indices, axis=(1, 2))
    
    # Rate the possible source-sink pairs by their proximity to existing anomalies
    proposed_ex_line = voltage_all_possible[np.argsort(proximity_to_boundary)[::-1]][:n_of_voltages]
    ex_line_start = 0
    ex_line_end = 2 # From 1 to desired number of current pairs
    
    while(True):
        new_ex_lines = proposed_ex_line[ex_line_start:ex_line_end] # Select a number of the best current pairs
        new_volt_mat, ind_temp = volt_matrix_generator(ex_mat=new_ex_lines, n_el=n_el, volt_pair_mode=volt_mode)
        # Generate Jacobian for voltage elements corresponding to new current pairs
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
        
        # Function call of propose_ex_volt_mat to obtain new voltage pairs for each new current pairs
        proposed_ex_volt_mat, repeated_ex = propose_ex_volt_mat(new_ex_lines,
                            ex_volt_mat, save_filename, total_map, n_of_voltages=n_of_voltages, counter=0, n_pix=n_pix)
        if (ex_line_end >= len(proposed_ex_line)):
            print("No new unique voltages proposed and no remaining current pairs; breaking loop ...")
        elif ((ex_line_end+ex_line_end <= len(proposed_ex_line)) and (len(proposed_ex_volt_mat)==0)):
            print("No new unique voltages proposed; relooping over new current pairs ...")
            ex_line_start = ex_line_start + ex_line_end
            ex_line_end = ex_line_end + ex_line_end
        else: # Return is only called in this else statement
            new_ex_lines = np.delete(new_ex_lines, repeated_ex, axis=0)  # deletion of any repeated current pairs
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
        
def adaptive_ESA(mesh_obj: None, volt_mat: None, ex_mat: np.ndarray, ind: None, voltages: np.ndarray, 
                      volt_mode:str='all', n_of_voltages: int=10, n_el: int=32, n_per_el: int=3, n_pix: int=64, pert: float=0.5, 
                      p_influence: float=-10., p_rec: float=10., p: float=0.5, lamb: float=0.1, do_plot: bool=True) -> np.ndarray:
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
    volt_mat = ex_volt_meas[:, 2:]
    print("\nSize of excitation matrix.")
    print("No. of current pairs: ", len(ex_mat))
    print("No. of voltage pairs: ", len(volt_mat))
    if (do_plot == True):
        plt.figure()
        im = plt.imshow(reconstruction, cmap=plt.cm.viridis, origin='lower', extent=[-1, 1, -1, 1])
        plt.title("GREIT Reconstruction\n(Current, Voltage) pairs: ("+str(len(ex_mat))+", "+str(len(volt_mat))+")")
        plt.colorbar(im)
        plt.savefig(filepath+"ESA reconstruction "+str(i))
        plt.show()
    return proposed_ex_volt_lines, ex_volt_meas, ex_mat, ind, reconstruction, total_map

"""
##############################################################################
 The following functions belong to the selections_algorithm.
     A module called within component_control, contains methods for the different selection algorithms for application to the
    measurement hardware.
    - Standard
    - GetNextElectrodes
    - GetNextElectrodesESA
##############################################################################
"""        
        
        
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

    if algorithm == 'Standard':
        try:
            if all_measurement_electrodes == None:
                all_measurement_electrodes = Standard(no_electrodes, **algorithm_parameters)
        except:
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
            ex_volt_mat, ex_mat, volt_mat, ind = initialise_ex_volt_mat(current_mode='opp', volt_mode='adj',n_el=32, ex_mat_length=32,
                                                                        ex_spaced=False)
            mesh_obj = mesh(n_el=n_el,start_pos='mid') # mid for our resistor grid since electrode 0 is at the centre of a side
            next_electrodes = ex_volt_mat
            ex_volt_index_data = (ex_volt_mat, ex_mat, volt_mat, ind, mesh_obj)
        else:
            #GetNextElectrodes(*algorithm_parameters)
            #voltages = np.random.random(len(ex_volt_mat)) # random voltages to ensure script working
            #QWARGS: adaptive_ESA(mesh_obj,volt_mat, ex_mat, ind, voltages, num_returned, n_el=32, do_plot=True):
            proposed_ex_volt_lines, ex_volt_meas, ex_mat, ind, reconstruction, total_map = adaptive_ESA(n_el=n_el, num_returned=10,
                                                                                                        **algorithm_parameters)
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
        # Adaptive_ESA function calls getNextPrediction_partialforwardsolver with some extra plotting functionality
        proposed_ex_volt_lines, ex_volt_meas, ex_mat, ind, sigma, total = adaptive_ESA(n_el=n_el, n_of_voltages=voltages_to_return,
                                                    volt_mode=ESA_volt_mode, mesh_obj=mesh_obj,volt_mat=volt_mat, ex_mat=ex_mat, 
                                                    ind=ind, voltages=voltages, p=p,n_pix=n_pix, lamb=lamb, do_plot=plot)
        volt_mat = ex_volt_meas[:, 2:]
        ex_volt_index_data = (ex_volt_meas, ex_mat, volt_mat, ind, mesh_obj)

    return proposed_ex_volt_lines, ex_volt_index_data


"""
##############################################################################
 The following functions belong to the component_control.py modules.
    The main pyVisa module which contain functions for controlling the switchbox and lock-in measurement operation.
    - SetMeasurementParameters
    - GetMeasurement
    - FlickSwitch
    - MapSwitches
    - ClearSwitches
    - wait_test
    - RunEIT
##############################################################################
""" 

# These are the setup parameters and addresses to connect the hardware to an open pyVisa session

#SET DEVICE IP AND PORT
lockin_ip = '169.254.147.1'     # Lock-in amplifier address
lockin_port = '1865'            # By default, the port is 1865 for the SR860.
lockin_lan_devicename = 'inst0' # By default, this is inst0. Check NI MAX

''' SWITCH USED IS ACCESSED VIA GPIB NOT LAN
switch_ip = '10.0.0.2' #default for CYTECH IF-6 module as used in VX256 is 10.0.0.2
switch_port = '23456' #default for CYTECH IF-6 module as used in VX256 is 23
'''
#SET SwITCHBOARD GPIB ADDRESS
switch_primary_address = '7'
switch_address = 'GPIB0::'+switch_primary_address+'::INSTR' # Create devices (resources) address strings
lockin_address = 'TCPIP::'+lockin_ip+'::'+lockin_lan_devicename+'::'+'INSTR'
rm = pyvisa.ResourceManager() # Create resource manager using py-visa backend ('@py') leave empty for NI VIS
#print available devices (resources)
print(rm.list_resources())
switch = rm.open_resource(switch_address) # Connect to devices
lockin = rm.open_resource(lockin_address)
switch.read_termination = '\n'            # Cytech manual says 'enter' so try \n, \r or combination of both
switch.write_termination = '\n'
#lockin.read_termination = '\f'           # SR860 manual says \lf so \f seems to be equivalent in python)
lockin.write_termination = '\f'
print("switch", switch.session)
print("lockin", lockin.session)
        
def SetMeasurementParameters(parameters):
    '''
    Assigns a parameter to data channel of SR860 lock-in for each parameter given in array of strings. 
    If fewer than 4 parameters are given, the remaining channels will not be changed from previous state.
    The parameter list is:
    i enumeration
    0 X
    1 Y
    2 R
    3 THeta
    4 IN1
    5 IN2
    6 IN3
    7 IN4
    8 XNOise
    9 YNOise
    10 OUT1
    11 OUT2
    12 PHAse 
    13 SAMp 
    14 LEV el 
    15 FInt 
    16 FExt
    
    Inputs
    ------
    parameters: list of str or int
        parameters desired to be measured

    Outputs
    ------
    None 
    '''
    if parameters == None:
        parameters = ["X","THeta","XNoise","FInt"]
    channel = 1
    for i in range(0, min(4, len(parameters))):
         #The CDSP j, param command assigns a parameter to data channel j. This is the same parameter assignment as pressing the [Config] key.
        lockin.write('CDSP DAT'+str(channel)+", "+str(parameters[i])) 
        channel += 1
    return

def GetMeasurement(parameters=None, param_set=True):
    '''
    Uses SNAPD? lockin-command to query the amplifier for values of the 4 measurement variables as previously defined in
    setMeasurementParameters().
    
    Inputs
    ------ 
    parameters: list of str 
        corresponding to parameters desired to be measured by lock-in SR860. If none, defaults to R, THeta, SAMp, FInt
    param_set: bool. 
        If true, set the parameters to be measured. If false, take measurement using previously set parameters (Speeds up measurement by ~0.03s)
    Outputs
    ------  
    measurement_array: NDarray
        Array of floats corresponding to mesaurment values in Volts, Hz or Degrees. Ordered in same order as specified in parameters.
    
    '''
    if param_set == True:
        SetMeasurementParameters(parameters) 
    measurement = lockin.query('SNAPD?')
    measurement_array = np.fromstring(measurement, sep=',')
    return measurement_array

def FlickSwitch(state, module, relay):
    '''
    Sends message to switchbox to change state of switch according to state given 
    by string ('on' or 'off') or int (0 or 1). Switch corresponds to relay within module.
    
    Notes: When switch.write is called in the python terminal console, the program returns an output. Does this output only indicate
    that the GPIB command has been sent, the command has been sent and recevied by the switchbox, or that it has been sent, received
    and fully executed before returning. (handshake protocol or something)
    Inputs
    ------  
    state: str or int
        State to change switch to 'on' (0) or 'off' (1).
    module: int
        Module number desired switch is in. 
    relay: int
        Relay(aka switch) number of desired switch within module.

    Outputs
    ------  
    None 
    '''

    if state == 1 or state=='on':
        state_str = "L"
    elif state == 0 or state=='off':
        state_str = "U"
    else:
        print("Must include switch state. 0(open) 1(closed)")
        return
    switch.write(state_str+str(relay)+" "+str(module))
    return 0

def MapSwitches(electrode, lockin_connection):
    '''
    This function maps the electrode number and the desired measurement terminal (I+,I-,V+,V-) to the
    corresponding relay and module numbers. The current is driven by the lockins AC oscillator channel.
    
    NOTE: This function is hardcoded for the 32x8 switchbox configuration using in the authors project.
    Please see details of the report or contact the authours if you are struggling to code your own variant.
    Inputs
    ------ 
    electrode: int
        Electrode number corresponding to numbering on output of switchbox.
    lockin_connection: str
        Relevant measurement terminal lock-in connection ("sin+" is 0,"sin-" is 1,"V+" is 2,"V-" is 3)

    Outputs
    ------ 
    module: int
        Module number corresponding to relay needed to connect electrode to lockin_connection
    relay: int
        Relay number within module needed to connect electrode to lockin_connection
    '''
    relay = electrode % 16
    module = ((electrode // 16) * 8) + lockin_connection
    return module, relay

def ClearSwitches():
    '''
    Opens all switch connections in switchbox
    '''
    switch.write('C')
    return

def wait_test(pos_1, pos_2, freq, interval, tc, n_measurements):
    """
    This function performs the relevant measurements to obtain plot data for the minimum delay time.
    This switches between pos_1 and pos_2 electrodes to simulate the act of switching between measurements,
    then it calls time.sleep(interval*(j+1)) to wait some time before querying the lock-in for a
    measurement. This function iteratively increases over the wait interval.
    
    Parameters
    ----------
    pos_1 : Array, size 4, of form [sin+,sin-,V+,V-]
        An array representing electrodes for a four terminal measurement. This pos1 does not matter
        it is just used to emulate switching from one configuration to another before testing the rise time.
    pos_2 : Array, size 4, of form [sin+,sin-,V+,V-]
        An array representing electrodes for a four terminal measurement.
    freq : float
        The frequency of the lock-in sinusoidal A.C. driving current.
    interval : float
        The interval, in seconds, between subsequent delay time measurements.
    tc : int
        Time constant setting for the lock-in. E.g. tc=6 is 1 ms time constant
    n_measurements : int
        Number of time measurments to take, each one increasing by the interval.

    Returns
    -------
    R_array : 1D numpy.array of floats
        List of subsequent resistance values.

    """
    
    lockin.write('OFLT '+str(tc))
    lockin.write('IRNG 3')
    SetMeasurementParameters(["X","Y","THeta","FInt"])
    lockin.write("ISRC 1")             # Set voltage unput to differential mode (A-B)
    lockin.write("SLVL 1")
    lockin.write("FREQ " + str(freq))  # Set frequency
    lockin.write("OFLT " + str(tc))    # Set time constant, 11 = 300ms, 12 = 1s
    lockin.write("PHAS 0")             # set phase offset to 0
    lockin.write("SCAL 8")
    lockin.write("IRNG 3")
    
    Rin = 100e3  # Shunt resistor resistance
    V_in = 1     # V_rms for sinusoidal signal generator of lock-in
    I = V_in/Rin   
    measrument_pos_1 = setup_electrodes(pos_1, interval)
    R_pos_1 = measrument_pos_1[0]/I
    ClearSwitches()
    R_array = []  
    
    for j in range(0, n_measurements-1):
        ClearSwitches()
        for i in range(0,4):
            module, relay = MapSwitches(electrode=pos_2[i], lockin_connection=i)
            FlickSwitch(1, module, relay)
        time.sleep(interval*(j+1))
        measurement = lockin.query('SNAPD?')
        print('ILVL', lockin.query('ILVL?'))
        measurement_array = np.fromstring(measurement, sep=',')
        R = measurement_array[0]/I
        R_array.append(R)
    
    print('tc:', lockin.query('OFLT?'))
    return R_array

def RunEIT(algorithm='Standard', no_electrodes=32, max_measurements=10000, measurement_electrodes = None, 
            print_status=True, voltage=1, freq=30, wait=60, tc=12, **algorithm_parameters):
    '''
    This is an over-arching function which can be called to perform all of the measurements for the chosen electrode
    selection algorithm.
    
    Inputs
    ------ 
    algorithm: str
        Specifies electrode selection agolrithm. eg 'Standard' for adj-adj or 'Random' for random electrode placements. 
    no_electrodes: int
        Number of electrodes attached to sample
    max_measurements: int
        Maximum voltage measurements to be taken
    measurement_electrodes: NDarray
        A 4*N array of electrode positions for all measurements. Allows user to pre-generate desired electrode positions instead of using algorithm.
        Helps to speed up when using Standard algorithm.
    voltage: float
        Voltage of lock-in driving signal in Volts rms. Default 2V.
    freq: int
        Frequency of lock-in driving signal in Hz. Default 30Hz.
    tc: int
        Time constant used by lock-in amplifer. Corresponds to OFLT command fpr SR865 lock-in.
        0->1us, 1->3us, 2->10us, 3->30us, 4->100us, 5->300us,... 20->10ks, 21->30ks.
        Default 12->1s.
    wait: int    
        Time to wait between measurements divided by (1/f) of driving frequency ie no. of periods.
        Default 60, ie 2s for 30Hz.  
    print_status: bool
        Sets whether to print status messages
    algorithm_parameters: **kwargs
        Allows user to pass relevant parameters of desired algorithm
    
    Outputs
    ------ 
    v_difference: NDarray
        1*N float array of all voltage measurements taken
    flick_times_np: NDarray 
        Float array of all time durations during which a switch command was executed
    get_times_np: NDarray
        Float array of all time durations during which a lock-in command was executed
    '''
    
    print("starting EIT...")
    start = time.time()
    print("Initialising lock-in....")
    SetMeasurementParameters(["X","Y","THeta","FInt"])
    lockin.write("ISRC 1")               # Set voltage unput to differential mode (A-B)
    lockin.write("SLVL " + str(voltage))
    lockin.write("FREQ " + str(freq))    # frequency
    lockin.write("OFLT " + str(tc))      # time constant 11 = 300ms, 12 = 1s
    lockin.write("PHAS 0")               # set phase offset to 0
    lockin.write("SCAL 8")
    lockin.write("IRNG 3")
    print("Waiting...")
    time.sleep(2)
    '''
    print("Lock-in test measurement", lockin.query("SNAPD?"))
    '''
    print("Clearing switches...")
    ClearSwitches()
    print("Switches cleared")
    #print("Switch status", switch.query('S'))
    v_diff = []
    electrode_posns =[]
    flick_times = []
    get_times = []
    ex_volt_index = None # for ESA
    keep_measuring = True
    y_diff=[]
    
    while keep_measuring == True:
        measurement = -1
        for k in range(0,max_measurements):
            measurement = measurement+1
            if algorithm == 'ESA':
                current_mode='opp' # for initial ex mat
                volt_mode='adj' # for initial volt mat
                ESA_volt_mode = 'all' # for availalbe volt pairs in ESA
                current_pairs_to_return = 2 #(REDUNDANT: SEE EX_LINE_END IN get_next_preidcition.py) no. of current pairs to propose, 1-3 recommended
                voltages_to_return = 50 # no. of voltages per current pairs, 10 to 50 recommended
                n_pix=64
                pert=0.5
                p_influence=-10.
                p_rec=10.
                ex_mat_length = 10
                spaced_ex_mat = True
                p = 0.5
                lamb = 0.02
                ESA_params = (current_mode, volt_mode, ESA_volt_mode, current_pairs_to_return,
                              voltages_to_return, n_pix, pert, p_influence, p_rec, ex_mat_length, spaced_ex_mat, p, lamb)
                
                next_electrodes, ex_volt_index = GetNextElectrodesESA(voltages=v_diff, ESA_params=ESA_params,
                        ex_volt_index_data=ex_volt_index, n_el=no_electrodes, max_measurement=max_measurements, measurement=k)
                if(measurement==max_measurements):
                    keep_measuring=False
                                                
            # Form of ex_volt_index is shown below, this is for compatability with ESA looping. See get_next_predicition for details.
            #ex_volt_index = (ex_volt_meas, ex_mat, volt_mat, ind, mesh_obj)
            else:
                next_electrodes, keep_measuring = GetNextElectrodes(algorithm=algorithm, no_electrodes=no_electrodes,
                                                             measurement=k, all_measurement_electrodes = measurement_electrodes,
                                                             **algorithm_parameters)
            next_electrodes = next_electrodes.astype(np.int32)
            if keep_measuring == False:
                break
            start_clear = time.time()
            ClearSwitches()
            end_clear = time.time()
            clear_time = end_clear - start_clear
            flick_times.append(clear_time)
            try:
                next_electrodes_shape = (next_electrodes.shape[0],  next_electrodes.shape[1])
            except IndexError:
                next_electrodes_shape = (1, next_electrodes.shape[0])
            '''
            try: 
                print("next .shpae", next_electrodes.shape[1])
            except IndexError:
                print("index error")
            print("next electrode shape", next_electrodes_shape)
            '''
            
            for i in range(0, next_electrodes_shape[0]):
                
                ClearSwitches()
                
                for j in range(0, next_electrodes_shape[1]):
                    module = 0
                    relay = 0
                    try:
                        module, relay = MapSwitches(electrode=next_electrodes[i][j], lockin_connection=j)
                        #print("next_electrodes[j] ", next_electrodes[i])
                    except IndexError:
                        module, relay = MapSwitches(electrode=next_electrodes[j], lockin_connection=j)
                        #print("next_electrodes[j] ", next_electrodes[j])
                        
                    start_flick = time.time()
                    lockin_no = j
                    #electrode_no = next_electrodes[i][j]
                    #print("electrode no:", next_electrodes[i][j])
                    #print("module:", module, "Expected:", ((electrode_no // 16) * 8)+ lockin_no)
                    #print("relay:", relay, "Expected:",electrode_no % 16)
                    #print("module:", module)
                    #print("relay:", relay)
                    FlickSwitch('on', module, relay)
                    #if measurement == 0 and j==0:
                        #time.sleep((1/freq)*wait*1) # critical time between 0.1 and 0.5
                    end_flick = time.time()
                    flick_times.append(end_flick - start_flick)
                start_get =time.time()
                
                print_switch_status=False 
                if print_status==True:
                    print("Measurement: "+str(i)+", next electrode: "+str(next_electrodes[i])+", keep measuring: "+str(keep_measuring))
                    if print_switch_status==True:
                        switch_status = switch.query('S')
                        print(switch_status)
                
                time.sleep(wait * (1/freq)) # Delay time to let lockin settle down 
                x=0
                y=0
                theta=0
                fint=0
                x, y, theta, fint = GetMeasurement(param_set=False)
                #x = x
                end_get = time.time()
                get_time = end_get - start_get
                get_times.append(get_time)
                if algorithm == 'ESA':
                    scaling_factor = 1e5
                else:
                    scaling_factor = 1
                v_diff.append(x*scaling_factor)
                y_diff.append(y)
                try:
                    electrode_posns.append(next_electrodes[i,:])
                except IndexError:
                    electrode_posns.append(next_electrodes)
        
        v_difference = np.array(v_diff)
        y_diff = np.array(y_diff)
        electrode_positions = np.array(electrode_posns)
        flick_times_np = np.array(flick_times)
        get_times_np = np.array(get_times)
        break
    ClearSwitches() 
    if algorithm == 'ESA':
        try:
            mesh_obj = ex_volt_index[4]
        except:
            mesh_obj = None
    else:
        mesh_obj = None
    
    end = time.time()
    duration = end - start
    no_voltages = len(v_difference)
    average_time = duration / no_voltages
    print("EIT finished")
    #print("Voltages: ", v_difference)
    #print("Positions:", electrode_positions)
    print(str(no_voltages)+" measurements taken in "+str(duration)+" seconds.")
    print("Average time for measurement: ", average_time)
    total_switch_time = np.sum(flick_times_np)
    average_switch_time = np.mean(flick_times_np)
    print("Switch commands: ", len(flick_times_np))
    print("Total switch time", total_switch_time)
    print("Average switch time", average_switch_time)
    total_lockin_time = np.sum(get_times_np)
    average_lockin_time = np.mean(get_times_np)
    print("Lock-in commands: ", len(get_times_np))
    print("Total lock-in time", total_lockin_time)
    print("Average lock-in time", average_lockin_time)

    return v_difference, electrode_positions, mesh_obj, flick_times_np, get_times_np, y_diff
