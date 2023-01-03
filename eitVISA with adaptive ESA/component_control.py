# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 10:28:08 2020
Authors: Adam Coxson and Frederik Brookebarnes, MPhys Undergrads, The University of Manchester
Project: Automated Electrical Impedance Tomography of Graphene
Module:  component_control
Dependancies: pyVisa package, selection_algorithms.py

This script acts as a control interface between a remote controlled desktop and a
lock-in amplifier and Cytec switchbox used to measure voltage across samples.
This uses GPIB commands via the pyVisa API. This takes four-terminal current and
voltage measurements for electrical impedance tomography setups. This is integrated
with the pyEIT package: https://github.com/liubenyuan/pyEIT

The RunEIT fuction can be used to run a number of different electrode selection algorithms
including an adaptive ESA which can select for potential current and voltage excitation pairs
which will provide the best improvement of the conductivity map.


------ FUNCTION DEFINITIONS -------
- TimeStamp(fname, fmt='%Y-%m-%d-%H-%M-%S_{fname}')
- SaveEIT(positions, voltages, filename):
- TwoPointIndices(no_electrodes)
- SetMeasurementParameters(parameters)
- GetMeasurement(parameters=None, param_set=True)
- FlickSwitch(state, module, relay)
- MapSwitches(electrode, lockin_connection)
- ClearSwitches()
- eit_scan_lines(ne=16, dist=1)
- RunEIT(algorithm='Standard', no_electrodes=32, max_measurements=10000, measurement_electrodes = None, 
            print_status=True, voltage=2, freq=30, wait=60, tc=12, **algorithm_parameters)
"""

import pyvisa
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import time
import os
import math
from datetime import datetime
from selection_algorithms import eit_scan_lines, voltage_meter, Standard, GetNextElectrodes, GetNextElectrodesAESA
from selection_algorithms import van_der_Pauw_selection, vdp_newton_raphson_iter, van_der_pauw_calculations
from aesa_iteration import reconstruction_plot, initialise_ex_volt_mat, ex_mat_ind_finder
import matplotlib as mpl
from meshing import mesh

import pyeit.eit.jac as jac
import greit_rec_training_set as train
from pyeit.eit.fem_for_given_meas import Forward as Forward_given
from pyeit.eit.interp2d import sim2pts
                    

#SET DEVICE IP AND PORT
lockin_ip = '169.254.147.2' # lock-in amplifier address
lockin_port = '1865'  # By default, the port is 1865 for the SR860.
lockin_lan_devicename = 'inst0' # By default, this is inst0. Check NI MAX

''' SWITCH USED IS ACCESSED VIA GPIB NOT LAN
switch_ip = '10.0.0.2' #default for CYTECH IF-6 module as used in VX256 is 10.0.0.2
switch_port = '23456' #default for CYTECH IF-6 module as used in VX256 is 23
'''
#SET DEVICE ADDRESSES
switch_primary_address = '7'
switch_address = 'GPIB0::'+switch_primary_address+'::INSTR'
lockin_address = 'TCPIP::'+lockin_ip+'::'+lockin_lan_devicename+'::'+'INSTR'
rm = pyvisa.ResourceManager() # create resource manager using py-visa backend ('@py') leave empty for NI VIS
print(rm.list_resources())
switch = rm.open_resource(switch_address) #connect to devices
lockin = rm.open_resource(lockin_address)
switch.read_termination = '\n' # cytech manual says 'enter' so try \n, \r or combination of both
switch.write_termination = '\n'
#lockin.read_termination = '\f' # SR860 manual says \lf so \f seems to be equivalent in python)
lockin.write_termination = '\f'
print("switch", switch.session)
print("lockin", lockin.session)
#lockin.flush()


def TimeStamp(fname, fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
    """
    Gets current time stamp for datafile saving.
    """
    return datetime.now().strftime(fmt).format(fname=fname)

def SaveEIT(positions, voltages, file):
    """
    Saves csv data of measurements obtained from hardware operation within RunEIT.

    Parameters
    ----------
    positions : np.ndarray (M,4) with each row [current source, current sink, voltage+, voltage-]
        The concatenated current pair and voltage pair matrix.
    voltages : 1D array of float
        List of voltage measurements from system. These are either real measurements or simulated.
    filename : string
        Filename written to local directory or specified \\data\\ folder.

    Returns
    -------
    filename

    """
    print("Saving...")
    file = TimeStamp(file+".csv")
    filename = os.getcwd() + "\\data\\"+file
    filename = file
    #try:
        #os.mkdir(filepath)
    #except:
        #print("Error making file:",filename)
        #return None
            
    data = [positions[:,0], positions[:,1], positions[:,2], positions[:,3], voltages]
    data = np.asarray(data).T
    np.savetxt(filename, data, fmt=['%i', '%i', '%i', '%i', '%e'], delimiter=",", header="sin+,sin-,v_high,v_low,voltage", comments="")
    print("File saved as", str(filename))
    return filename

def SaveEIT_means_std(positions, mean_voltages, std_voltages, I=1e-5, file='default'):
    """
    Saves csv data of measurements obtained from hardware operation within RunEIT. This is the improved version of SaveEIT function.
    This is for averaged voltage measurements repeated at separations of several periods. The std data is unused within the EIT analysis
    but is useful to indicate measurement consistency. Aim for less than 10% error, or ideally 1% or less.
    
    To Do: Create a merged form of SaveEIT, SaveEIT_means_std, and SaveEIT_x_y, etc.
    
    Parameters
    ----------
    positions : np.ndarray (M,4) with each row [current source, current sink, voltage+, voltage-]
        The concatenated current pair and voltage pair matrix.
    mean_voltages : 1D array of float
        List of averaged voltage measurements. These are either real measurements or simulated.
    std_voltages : 1D array of float
        List of std deviation from averaged voltage measurements.
    I : float, optional
        Input current from lockin voltage generator and resistor. The default is 1e-5 Amps.
    file : string
        Filename written to local directory or specified \\data\\ folder.

    Returns
    -------
    filename and writes CSV to working directing data folder

    """

    print("Saving...")
    file = TimeStamp(file+".csv")
    filename = os.getcwd() + "\\data\\" + file
    filename = file
    data = [positions[:,0], positions[:,1], positions[:,2], positions[:,3], mean_voltages, std_voltages, mean_voltages/I, std_voltages/I]
    data = np.asarray(data).T
    np.savetxt(filename, data, fmt=['%i', '%i', '%i', '%i', '%e', '%e','%e', '%e'], delimiter=","
               , header="sin+,sin-,v high,v low,mean V,std, mean R,std", comments="")
    print("File saved as", str(filename))
    return filename

def SaveEIT_x_y(positions, x, x_err, y, y_err, I=1e-5, file='default'):

    print("Saving...")
    file = TimeStamp(file+".csv")
    filename = os.getcwd() + "\\data\\" + file
    filename = file
    data = [positions[:,0], positions[:,1], positions[:,2], positions[:,3], x, x_err, y, y_err, x/I, x_err/I]
    data = np.asarray(data).T
    np.savetxt(filename, data, fmt=['%i', '%i', '%i', '%i', '%e', '%e','%e', '%e', '%e','%e'], delimiter=","
               , header="sin+,sin-,v high,v low,mean V,std, mean Y, std, mean R,std", comments="")
    print("File saved as", str(filename))
    return filename

def read_data_EIT(filename):
    positions = np.loadtxt(filename, delimiter=",", skiprows=1, usecols=[0,1,2,3])
    voltages = np.loadtxt(filename, delimiter=",", skiprows=1, usecols=4)

    return voltages, positions

def TwoPointIndices(n_el:int=64):
    """
    Returns a list of electrodes that correspond to intercontact measurements such that the positive I and V terminals
    are both on the same electrode, i.e. [I+,I-,V+,V-] = [0,1,0,1], [1,2,1,2], so on. Used to check the contacts and
    calculate the intercontact resistances.

    Parameters
    ----------
    n_el : int, optional
        The number of electrodes around sample. The default is 64.

    Returns
    -------
    posns : np.ndarray (M,4) with each row [current source, current sink, voltage+, voltage-]
        The concatenated current pair and voltage pair matrix.
    """
    posns = np.zeros((n_el, 4))
    for j in range(0, n_el):
        posns[j,0] = j
        posns[j,1] = (j+1) % n_el
        posns[j,2] = (j) % n_el
        posns[j,3] = (j+1) % n_el
    return posns

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
        parameters = ["X","Y","XNoise","FInt"]
    channel = 1
    for i in range(0, min(4, len(parameters))):
         #The CDSP j, param command assigns a parameter to data channel j. This is the same parameter assignment as pressing the [Config] key.
        lockin.write('CDSP DAT'+str(channel)+", "+str(parameters[i])) 
        channel += 1
    return

def GetMeasurement(freq, repeats=5, n_periods=5, parameters=None, param_set=True):
    """
    Uses SNAPD? lockin-command to query the amplifier for values of the 4 measurement variables as previously defined in
    setMeasurementParameters(). Repeats measurements separated by a number of periods for avg and std.
    
    Inputs
    ------ 
    parameters: list of str 
        corresponding to parameters desired to be measured by lock-in SR860. If none, defaults to R, THeta, SAMp, FInt
    param_set: bool. 
        If true, set the parameters to be measured. If false, take measurement using previously set parameters (Speeds up measurement by ~0.03s)
    Outputs

    Parameters
    ----------
    freq : float
        Internal frequency of lockin oscillator, required for measurement repeat delay.
    repeats : int, optional
        Number of repeated measurements. The default is 5.
    n_periods : int, optional
        number of periods to delay between repeated measures. The default is 5.
    parameters: list of str 
        corresponding to parameters desired to be measured by lock-in SR860. If none, defaults to R, THeta, SAMp, FInt
    param_set: bool. 
        If true, set the parameters to be measured. If false, take measurement using previously set parameters (Speeds up measurement by ~0.03s)

    Returns
    -------
    X, X_std: 1D array of float
        Means and deviations of x channel measurement (i.e. x voltage)
    Y, Y_std: 1D array of float
        Means and deviations of y channel measurement (i.e. y voltage)
    """
    x = np.zeros(repeats)
    y = np.zeros(repeats)
    if param_set == True:
        SetMeasurementParameters(parameters)
        
    for i in range(0,repeats):
        time.sleep(1/freq * n_periods) # Delay between repeated measures
        measurement = lockin.query('SNAPD?')
        measurement_array = np.fromstring(measurement, sep=',')
        x[i] = measurement_array[0]
        y[i] = measurement_array[1]
        
    X = np.mean(x)
    X_std = np.std(x)
    Y_std = np.std(y)
    Y = np.mean(y)
    return X, X_std, Y, Y_std

def query_lockin_settings():
    """
    Prints out current settings of the lockin for comparison to manual. 
    """
    settings = ['TBMODE?','TBSTAT?','PHAS?','FREQ?','FREQINT?','FREQEXT?','FREQDET?','HARM?',
    'HARMDUAL?','BLADESLOTS?','BLADEPHASE?','SLVL?','SOFF?','REFM?','RSRC?','RTRG?',
    'REFZ?','IVMD?','ISRC?','ICPL?','IGND?','IRNG?','ICUR?',
    'ILVL?','SCAL?','OFLT?','OFSL?','SYNC?','ADVFILT?','ENBW?']
    
    for i in range(0,len(settings)):
        string = lockin.query(settings[i])
        print(settings[i],",",string)
    
    return None

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
    
    NOTE: This function is hardcoded for either the 32x8 or 64x4 switchbox configuration used in the project.
    Please see details of the reports or contact the authours if you are struggling to code your own variant.
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
    relay = int(electrode) % 16 # Valid for both 32x8 and 64x4
    
    #module = ((electrode // 16) * 8) + lockin_connection # FOR 32 x 8
    
    module_list = [[0, 7, 8, 15],[1, 6, 9, 14],[2, 5, 10, 13],[3, 4, 11, 12]] # FOR 64 x 4
    module = module_list[int(lockin_connection)][int(electrode) // 16]
    
    return module, relay

def ClearSwitches(maintain_ground:bool=True, n_el:int=64):
    '''
    Opens all switch connections in switchbox. If ground must be maintained,
    also opens all other switches while keeping electrode 0 connected to ground.
    '''
    if maintain_ground == False:
        switch.write('C')
    else:
        module, relay = MapSwitches(electrode=0, lockin_connection=1)
        FlickSwitch('on', module, relay)
        for i in [0,2,3]:
            module, relay = MapSwitches(electrode=0, lockin_connection=i)
            FlickSwitch('off', module, relay)
        for j in range(1,n_el):
            for i in range(0,4):
                module, relay = MapSwitches(electrode=j, lockin_connection=i)
                FlickSwitch('off', module, relay)
    return None


def setup_electrodes(el_list, wait, maintain_ground=False, V_in=1):
    """
    Can be used to setup a single set of electrodes and return a measurement. Useful for debugging.
    Requires lockin and switch resource managers to be initialised
    Parameters
    ----------
    el_list : list, size 4, of form [sin+,sin-,V+,V-]
        An array representing electrodes for a four terminal measurement.
    wait : int or float
        The time, in seconds, the lockin will wait before taking a measurement after switches have been opened.
    Returns
    -------
    measurement_array : array, size 4
        The 4 measurement parameters the lockin is set to query upon SNAPD? command.

    """
    Rin = 10e3
    I = V_in/Rin
    repeats = 5
    freq = float(lockin.query("FREQ?"))
    voltages = np.zeros(repeats)
    ClearSwitches(maintain_ground=maintain_ground, n_el=64)
    for i in range(0,4):
        module, relay = MapSwitches(electrode=el_list[i], lockin_connection=i)
        FlickSwitch(1, module, relay)
        #print("Module:", module,"relay:",relay)
    if el_list[1] != 0:
        FlickSwitch('off',module=1,relay=0)
    #print(switch.query('S'))
    print("Electrodes", el_list)
    time.sleep(wait)
    for i in range(0,repeats):
        time.sleep(1/freq * 10)
        measurement = lockin.query('SNAPD?')
        measurement_array = np.fromstring(measurement, sep=',')
        voltages[i] = measurement_array[0]
        
    measurement_array = np.fromstring(measurement, sep=',')
    V = np.mean(voltages)
    V_err = np.std(voltages)
    # R = measurement_array[0]/I
    print(voltages)
    print("V:",V,"±",V_err,"R:",V/I,"±",V_err/I)

    return measurement_array
    #return None


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
    lockin.write("ISRC 1") #Set voltage unput to differential mode (A-B)
    lockin.write("SLVL 1")
    lockin.write("FREQ " + str(freq))     # frequency
    lockin.write("OFLT " + str(tc)) # time constant 11 = 300ms, 12 = 1s
    lockin.write("PHAS 0") # set phase offset to 0
    #lockin.write("ASCL")   # autoscale
    lockin.write("SCAL 8")
    lockin.write("IRNG 3")
    
    Rin = 100e3
    V_in = 1
    I = V_in/Rin   
    measrument_pos_1 = setup_electrodes(pos_1, interval)
    R_pos_1 = measrument_pos_1[0]/I
    ClearSwitches(maintain_ground=False)
    R_array = []  
    
    for j in range(0, n_measurements-1):
        ClearSwitches(maintain_ground=False)
        for i in range(0,4):
            module, relay = MapSwitches(electrode=pos_2[i], lockin_connection=i)
            FlickSwitch(1, module, relay)
            #print("Module:", module,"relay:",relay) 
        time.sleep(interval*(j+1))
        measurement = lockin.query('SNAPD?')
        print('ILVL', lockin.query('ILVL?'))
        measurement_array = np.fromstring(measurement, sep=',')
        R = measurement_array[0]/I
        R_array.append(R)
        #print("R:",R)
    #print("R_pos_1:", R_pos_1)
    #print("R_array:", R_array)
    
    print('tc:', lockin.query('OFLT?'))
    return R_array

def open_next_switches(new_electrodes, old_electrodes):
    """
    Opens switches while maintaining a ground connection at all times

    Parameters
    ----------
    new_electrodes : list or array
        list of 4 electrodes to be opened for next measurement.
    old_electrodes : list or array
        List of 4 electrodes from previous measurement so the correct switches are closed.
    """
    
    for i in [0, 2, 3]: # Open previous electrodes not connected to ground (I- terminal_
        module, relay = MapSwitches(electrode=old_electrodes[i], lockin_connection=i)
        FlickSwitch('off', module, relay)
        
    # Connect the new ground, now two switches are connected to the ground terminal
    module, relay = MapSwitches(electrode=new_electrodes[1], lockin_connection=1)
    FlickSwitch('on', module, relay)
    
    if (new_electrodes[1]!=old_electrodes[1]): # If the new and old ground terminals are different, close old ground switch
            module = 0
            relay = 0
            module, relay = MapSwitches(electrode=old_electrodes[1], lockin_connection=1)
            FlickSwitch('off', module, relay)

    for i in [0, 2, 3]: # Open the remaining new electrodes.
        module, relay = MapSwitches(electrode=new_electrodes[i], lockin_connection=i)
        FlickSwitch('on', module, relay)
    return None

def vdp_switches_shorted(new_electrodes, n_el=64):
    """
    For a typical EIT sample setup, there are no corner contacts, only edge ones offset
    by some amount as per the mesh. For van der pauw method, can short the two adjacent edge
    electrodes over the corner to act as one electrode. This function is the equivalent
    of open_switches() but for 8 electrodes in total, 2 per corner.

    Parameters
    ----------
    new_electrodes : 2D list of 2 electrodes per column
        The lsit of corner electrodes for shorting.
    n_el : int, optional
        Number of electrodes. The default is 64.
    Returns
    -------
    None.
    """
    
    ClearSwitches(maintain_ground=False, n_el=n_el)
    
    for i in [0,1]: # Activate two ground terminals first
        module, relay = MapSwitches(electrode=new_electrodes[1][i], lockin_connection=1)
        FlickSwitch('on', module, relay)
    
    for j in [0,2,3]:
        for i in [0,1]:
            module, relay = MapSwitches(electrode=new_electrodes[j][i], lockin_connection=j)
            FlickSwitch('on', module, relay)
    return None
    

def RunEIT(algorithm='Standard', no_electrodes=64, max_measurements=10000, measurement_electrodes=None, 
            print_status=True, voltage=1, freq=30, wait=60, tc=12, AESA_params=None, **algorithm_parameters):
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
        Allows user to pass relevant parameters of desired algorithm, redundant

    Outputs
np.array(x_diff), np.array(x_err), np.array(y_diff), np.array(y_err), np.array(electrode_posns), mesh_obj
    x_diff: 1D array
        list of mean voltage measurements
    x_err: 1D array
        List of errors on mean voltage measurements
    y_diff: 1D array
        list of mean y-channel measurements
    y_err: 1D array
        List of errors on mean y-channel measurement
    electrode_posns: ND np.array (len(voltages), 4)
        Electrode positions corresponding to measured voltages
    mesh_obj: class object
        The mesh used in the AESA (if set to AESA)
        
    '''
    
    print("Starting EIT")
    start = time.time()
    SetMeasurementParameters(["X","Y","THeta","FInt"])
    lockin.write("ISRC 1") # Set voltage unput to differential mode (A-B)
    lockin.write("SLVL " + str(voltage))
    lockin.write("FREQ " + str(freq))     # frequency
    lockin.write("OFLT " + str(tc)) # time constant 11 = 300ms, 12 = 1s
    lockin.write("PHAS 0") # set phase offset to 0
    lockin.write("OFSL 2")
    lockin.write("SCAL 18")
    lockin.write("IRNG 4")
    time.sleep(0.5)
    ClearSwitches(maintain_ground=True, n_el=no_electrodes)
    print("Switches cleared")
    #print("Switch status", switch.query('S'))  
    x_diff = []
    x_err=[]
    y_diff = []
    y_err=[]
    electrode_posns = []
    ex_volt_index = None # for ESA
    old_electrodes = [0,0,0,0]
    
    #for k in range(0, len(measurement_electrodes)):
    for k in range(0, max_measurements):
        if algorithm == 'ESA':
            proposed_electrodes, ex_volt_index = GetNextElectrodesAESA(voltages=x_diff, ESA_params=AESA_params,
                    ex_volt_index_data=ex_volt_index, n_el=no_electrodes, max_measurement=max_measurements, measurement=k)
            proposed_electrodes = proposed_electrodes.astype(np.int32)
            for i in range(len(proposed_electrodes)):
                next_electrodes = proposed_electrodes[i]
                open_next_switches(new_electrodes=next_electrodes, old_electrodes=old_electrodes)
                old_electrodes = next_electrodes
            
                print_switch_status=False
                if print_status==True:
                    print("Loop "+str(k)+", measurement "+str(i)+", Electrodes: "+str(next_electrodes))
                    if print_switch_status==True:
                        switch_status = switch.query('S')
                        print(switch_status)
                        
                time.sleep(wait * (1/freq)) 
                x, x_std, y, y_std = GetMeasurement(freq=freq, repeats=10,n_periods=10,param_set=False)
                x_diff.append(x)
                x_err.append(x_std)
                y_diff.append(y)
                y_err.append(y_std)
                electrode_posns.append(next_electrodes)
            
        else:
            #next_electrodes, keep_measuring = GetNextElectrodes(algorithm=algorithm, no_electrodes=no_electrodes,
                                                         #measurement=k, all_measurement_electrodes=measurement_electrodes, **algorithm_parameters)
            
            next_electrodes = measurement_electrodes[k].astype(np.int32)
            #vdp_switches_shorted(new_electrodes=next_electrodes, n_el=no_electrodes) # for shorted corner vdp measurements only
            open_next_switches(new_electrodes=next_electrodes, old_electrodes=old_electrodes)
            old_electrodes = next_electrodes
        
            print_switch_status=False
            if print_status==True:
                print("Measurement: "+str(k)+", next electrodes: "+str(next_electrodes))
                if print_switch_status==True:
                    switch_status = switch.query('S')
                    print(switch_status)
                    
            time.sleep(wait * (1/freq)) 
            x, x_std, y, y_std = GetMeasurement(freq=freq, repeats=10,n_periods=10,param_set=False)
            x_diff.append(x)
            x_err.append(x_std)
            y_diff.append(y)
            y_err.append(y_std)
            electrode_posns.append(next_electrodes)
            

            
            if k == len(measurement_electrodes)-1:
                ClearSwitches(maintain_ground=True, n_el=no_electrodes)
                break

    ClearSwitches(maintain_ground=True, n_el=no_electrodes)
    if algorithm == 'ESA':
        try:
            mesh_obj = ex_volt_index[4]
        except:
            mesh_obj = None
    else:
        mesh_obj = None
    end = time.time()
    duration = end - start
    no_voltages = len(x_diff)
    average_time = duration / no_voltages
    print("EIT finished")
    print(str(no_voltages)+" measurements taken in "+str(duration)+" seconds.")
    print("Average time for measurement: ", average_time)

    return np.array(x_diff), np.array(x_err), np.array(y_diff), np.array(y_err), np.array(electrode_posns), mesh_obj
    
def noise_check(positions, freqs, tcs, waits, reps):
    
    Rin = 100e3
    V_in = 1
    I = V_in/Rin
    R = np.zeros((np.shape(positions)[0], reps))
    for i in range(0, len(freqs)):
        for j in range(0, len(tcs)):
            for k in range(0, len(waits)):
                for l in range(0,reps):
                    voltages, positions, mesh, switch_times, lockin_times, y = RunEIT(no_electrodes=32, voltage=V_in, algorithm='Standard',measurement_electrodes=positions, 
                                                                freq=freqs[i],tc=tcs[j],wait=waits[k])
                    R[:,l] = voltages/I
                    #print(R[:,l])
                for m in range(0, np.shape(positions)[0]): 
                    R_mean = np.mean(R[m,:])
                    R_std = np.std(R[m,:])
                    R_std_per = (R_std/R_mean) * 100
                    print("Positions:", positions[m,:])
                    print("R:", R_mean, "±", R_std_per, "%")
    return positions, R

if  __name__ == "__main__":
    
    """
    This is the code to setup the hardware, choose relevant EIT algorithms, take measurements and save for EIT analysis or
    call EIT reconstruction functions for instant results.
    
    The main is split up  using If-else into different types of operation:
    -   Testing is used to take non-standard measurements, useful for trying to debug the setup. Often requires bits of code to be added
        to do the specific thing you want it to do.
    -   Standard. Used for standard/common EIT algorithms such as the adjacent-adjacent or opposite-adjacent stimulation patterns.
        The Electrode Selection Algorithm must be defined beforehand in electrode array definition.
    -   AESA. Used for the Machine Learning Adaptive ESA. Requires many more input parameters, also requires an initial set of measurements
        to be predefined in electrode array definition to before the Adaptive part of the algorithm can kick in. See literature for more details.
    -   Van der Pauw. Enables VdP measurements of the sample, if it is square. There are 3 different methods, shorted, narrow and wide. Shorted 
        corner method is the combination of narrow and wide methods, use this as default. Currently requires some uncommenting/commenting of the 
        RunEIT function to use the shorted method, as the positions variable is incompatible, ask the authour for more details on how to do this.
    -   Repeats and averaging. Same as the standard option but repeats the whole process to obtain averaging. This is mostly redundant with the
        current 'GetMeasurement' function as this performs repeats and averaging under-the-hood upon each measurement. However, possible still 
        useful for testing and averaging explicitly taken measurements that do not use the function groupings.
    -   Wait test. Slightly outdated code, this was used to determine the nescessary delay for the lockin to cycle over several time constants
        before consistent measurements could be taken.
    """
    
    Rin = 10e3  # Shunt resistor in Ohms
    V_in = 1     # V_rms in Volts for lock-in signal generator
    I = V_in/Rin # V and R result in a current of 10 uA
    
    freq = 8130            # Internal frequency of Lock-in AC signal generator
    timeconst = 6         # Time constant 3=30us, 4=100us, 5=300us, 6=1ms, 7=3ms, 8=10ms, 9=30ms,... 12=1s
    wait = 600             # Delay time before measurement to allow lock-in to cycle over several tcs
    algorithm = 'Standard' # ESA algorithm, either 'Standard' for normal, or 'ESA' for adaptive
    n_el = 64
    p = 0.5                # GREIT noise Covariance parameter
    lamb = 0.01            # GREIT Regularisation parameter
    current_mode = 'opp'   # For current and voltage pair initialisation
    volt_mode = 'adj'      # 'adj', 'opp', 'all'
    spaced_ex_mat = False  # Space out current pairs around perimeter for the initialisation, True for adaptive ESA
    ex_mat_length = 2
    query_lockin_settings()
    
    # dist for current, step for voltages
    #voltages, positions, mesh, switch_times, lockin_times, y = RunEIT(no_electrodes=32, algorithm='Standard', freq=8000,tc=6,wait=160,dist=1, step=1, print_status=False)
    #voltages, positions, mesh, switch_times, lockin_times, y = RunEIT(no_electrodes=32, algorithm='ESA', freq=8000,tc=6,wait=160, max_measurements=10)
   
    # ELECTRODE ARRAY DEFINITON - Initialising the list of contacts for a specific electrode selection algorithm.
    #two_test = np.array([[1,2,1,2],[0,3,0,3]])
    #el_array = TwoPointIndices(n_el)
    el_array, ex_mat, volt_mat, ind = initialise_ex_volt_mat(current_mode=current_mode, volt_mode=volt_mode, n_el=n_el, ex_spaced=spaced_ex_mat, ex_mat_length=None)

    #measurement_mode = 'testing'
    #measurement_mode = 'standard'
    #measurement_mode = 'aesa'
    #measurement_mode = 'van der pauw' 
    #measurement_mode = 'repeats and averaging'
    #measurement_mode = 'wait test'
    if measurement_mode == 'testing':
            freq = 8130            # Internal frequency of Lock-in AC signal generator
            timeconst = 6        # Time constant 3=30us, 4=100us, 5=300us, 6=1ms, 7=3ms, 8=10ms, 9=30ms,... 12=1s
            wait = 800 
            el_array = []
            for i in range(0,64):
                for j in range(0,64):
                    if i==j:
                        continue
                    else:
                        el_array.append([i,j,i,j])
                        
            x, x_err, y, y_err, positions, mesh = RunEIT(no_electrodes=n_el, voltage=V_in, algorithm=algorithm,
                       measurement_electrodes=np.array(el_array), freq=freq,tc=timeconst, wait=wait, print_status=False)
            SaveEIT_x_y(positions, x, x_err, y, y_err, I=I, file='x-y-contact-check')
            

         #electrodes = [[0,1,2,3], [4, 5, 6, 7], [32, 33, 34, 35], [48, 50, 52, 56]]
          #ClearSwitches(maintain_ground=True, n_el=n_el)
          # for el_list in el_array[:10]:
          #     print("")
          #     setup_electrodes(el_list=el_list, wait=(1/8000)*1000)
         
    elif measurement_mode == 'standard':
        # freq = 330           # Internal frequency of Lock-in AC signal generator
        # timeconst = 11       # Time constant 3=30us, 4=100us, 5=300us, 6=1ms, 7=3ms, 8=10ms, 9=30ms,... 12=1s
        # wait = 3000
        freq = 1330           # Internal frequency of Lock-in AC signal generator
        timeconst = 10       # Time constant 3=30us, 4=100us, 5=300us, 6=1ms, 7=3ms, 8=10ms, 9=30ms,... 12=1s
        wait = 4000 # wait in periods, 1/freq
        # freq = 8330           # Internal frequency of Lock-in AC signal generator
        # timeconst = 6       # Time constant 3=30us, 4=100us, 5=300us, 6=1ms, 7=3ms, 8=10ms, 9=30ms,... 12=1s
        # wait = 200
        voltages, v_err, y, y_err, positions, mesh = RunEIT(no_electrodes=n_el, voltage=V_in, algorithm='standard',
                       measurement_electrodes=el_array, freq=freq,tc=timeconst, wait=wait, print_status=True)
        #filename = "intercontact_big-cut_frq"+str(freq)+"_tc"+str(timeconst)+"_wait"+str(wait)
        #filename = "opp-adj_cut-big_frq"+str(freq)+"_tc"+str(timeconst)+"_wait"+str(wait)
        filename = "adj-adj_cut_frq"+str(freq)+"_tc"+str(timeconst)+"_wait"+str(wait)
        SaveEIT_means_std(positions, voltages, v_err, I=I,file=filename)
        
        # for i in range (0, len(voltages)):
        #     print(positions[i],"R:",voltages[i]/I,"±",(100 * v_err[i]/I)/(voltages[i]/I),"%")
        # mesh_params = [0.054, 3000, 9.34, 1.89]
       # mesh_obj = mesh(n_el=n_el, num_per_el=2, edge=0.08, el_width=0.04,ref_perm=10770, mesh_params=[0.045,2500,3.67,3.13]) # Making an empty mesh
       # reconstruction_plot(positions, voltages, mesh_obj=mesh_obj, ref_perm=10770, n_el=n_el, p=0.5, lamb=0.01, n_pix=128)
        
    elif measurement_mode == 'aesa':
        # AESA setup parameters
        # ref_perm=0.03352
        current_mode            = 'opp'
        volt_mode               = 'adj'
        ESA_volt_mode           = 'all'
        voltages_to_return      = 100
        n_pix                   = 128 
        cutoff                  = 0.916
        p_influence             = -20
        pert                    = 0.446 
        p_rec                   = 20 
        ex_mat_length           = 10
        spaced_ex_mat           = True
        p                       = 0.6
        lamb                    = 0.01
        #mesh_params             = [0.054,2500,10.,10.]
        mesh_params             = [0.045,2500,3.67,3.13]
        n_el                    = 64
        n_per_el                = 2
        AESA_params = (current_mode, volt_mode, ESA_volt_mode, voltages_to_return, n_pix, cutoff, pert, p_influence, p_rec,
                      ex_mat_length, spaced_ex_mat, p, lamb, mesh_params, n_el, n_per_el)
        freq = 1330           # Internal frequency of Lock-in AC signal generator
        timeconst = 10       # Time constant 3=30us, 4=100us, 5=300us, 6=1ms, 7=3ms, 8=10ms, 9=30ms,... 12=1s
        wait = 4000
        # freq = 8330           # debug settings
        # timeconst = 7       
        # wait = 1000
        voltages, v_err, y, y_err, positions, mesh = RunEIT(algorithm='ESA', no_electrodes=n_el, max_measurements=5, 
                                              voltage=V_in, freq=freq, wait=wait, tc=timeconst, AESA_params=AESA_params, print_status=True)
        
        #filename = "aesa_cut_2_p0-6_l0-01_npel2_finemesh"
       # SaveEIT_means_std(positions, voltages, v_err, I=I,file=filename)
        #reconstruction_plot(positions, voltages, mesh_obj=mesh, start_pos='left', n_el=64, n_per_el=n_per_el, p=0.6, lamb=0.01, n_pix=128)
        
        # lamb=0.01
        # p=0.65
        # volt_mat = positions[:, 2:]
        # ex_mat, ind = ex_mat_ind_finder(positions)
        # el_pos = np.arange(n_el * n_per_el).astype(np.int16)
        # fwd = Forward_given(mesh, el_pos, n_el)
        # f, meas, new_ind = fwd.solve_eit(volt_mat=volt_mat, new_ind=ind, ex_mat=ex_mat) # forward solve on the empty mesh
        # start_time = datetime.now()
        # eit_gn = jac.JAC(mesh, el_pos, ex_mat=ex_mat, f=f, perm=10770., parser='std')
        # # parameter tuning is needed for better EIT images
        # eit_gn.setup(p=p, lamb=lamb, method='kotre')
        # sigma_gn_raw = eit_gn.gn(voltages, maxiter=10, gtol=1e-4, p=p, lamb=lamb, method='kotre')
        # pts = mesh['node']
        # tri = mesh['element']
        # perm = mesh['perm']
        # sigma_gn = sim2pts(pts, tri, sigma_gn_raw)
        # fin_time = datetime.now()
        # print("Gauss-Newton time:", fin_time-start_time)                
        # #plt.figure(filename+"GN p="+str(p)+", lamb="+str(lamb))
        # plt.figure()
        # im_gn = plt.tripcolor(pts[:, 0], pts[:, 1], tri, np.real(sigma_gn), cmap=plt.cm.viridis)
        # cb=plt.colorbar(im_gn)
        # cb.ax.tick_params(labelsize=12)
        # plt.xlim(-1.,1.0)
        # plt.ylim(-1.,1.0)
        # plt.yticks(ticks=[-1.0,-0.75,-0.50,-0.25,0,0.25,0.50,0.75, 1.00], labels=[-1.0,-0.75,-0.50,-0.25,0,0.25,0.50,0.75, 1.00], fontsize=12)
        # plt.xticks(ticks=[-1.0,-0.75,-0.50,-0.25,0,0.25,0.50,0.75, 1.00], labels=[-1.0,-0.75,-0.50,-0.25,0,0.25,0.50,0.75, 1.00], fontsize=12)
        # #plt.title("Gauss-Newton p="+str(p)+", lamb="+str(lamb))
        # plt.show()
    

    # CODE TO REPEAT MEASUREMENTS TO OBTAIN MEANS AND DEVIATIONS
    elif measurement_mode == 'van der pauw':
        freq = 330           # Internal frequency of Lock-in AC signal generator
        timeconst = 11       # Time constant 3=30us, 4=100us, 5=300us, 6=1ms, 7=3ms, 8=10ms, 9=30ms,... 12=1s
        wait = 3000
        el_array = np.array(van_der_Pauw_selection(mode=3,n_el=64))
        R_vals = np.zeros(10)
        C_vals = np.zeros(10)
        for i in range(0,10):
            voltages, v_err, y, y_err, positions, mesh = RunEIT(no_electrodes=n_el, voltage=V_in, algorithm=algorithm,
                           measurement_electrodes=el_array, freq=freq,tc=timeconst, wait=wait, print_status=False)
    
            filename = "vdp-wide-cut_frq"+str(freq)+"_tc"+str(timeconst)+"_wait"+str(wait)
            
            R_sheet, sigma = van_der_pauw_calculations(abs(voltages)/I, current=I, thickness=25*(1e-6))
            #print("Sheet conductivity:",sigma)
            R_vals[i]=R_sheet
            C_vals[i]=sigma
            positions = np.array([[1,2,3,4]]*8)
            SaveEIT_means_std(positions, voltages, v_err, I=I,file=filename)
        print("R:",np.mean(R_vals),"±",np.std(R_vals))
        print("C:",np.mean(C_vals),"±",np.std(C_vals))
        
    elif measurement_mode == 'repeats and averaging':
        freq = 4000
        tc = 7
        wait = 600
        V_all = []
        for i in range(0,5):
            v, positions, mesh, y = RunEIT(no_electrodes=n_el, voltage=V_in, algorithm='Standard',
                      measurement_electrodes=el_array, freq=freq,tc=tc,wait=wait, print_status=False)
            V_all.append(v)
            
        V_all = np.asarray(V_all)
        V_mean = np.mean(V_all, axis=0)
        V_std = np.std(V_all, axis=0)
        R_all = V_all/I
        R_mean = np.mean(R_all,axis=0)
        R_std = np.std(R_all,axis=0)
        for i in range (0, len(R_mean)):
            print(positions[i],"V:",V_mean[i],"R:",R_mean[i],"±",R_std[i])

        # R_mean_string = np.array2string(R_mean, precision=8, separator=',',suppress_small=True)
        # R_std_string = np.array2string(R_std, precision=8, separator=',',suppress_small=True)
        # print(R_mean_string)
        # print(R_std_string)
        #filename = "step1-dist16-eit-8khz-tc=5-wait=25ms"
        filename = "intercontact_frq"+str(freq)+"_tc"+str(tc)+"_wait"+str(wait)
        SaveEIT_means_std(positions, V_mean, V_std, I=1e-5,file=filename)
        # R = voltages/I
        # for i in range(0,len(voltages)):
        #     print(positions[i],"R:",R[i], "V:",voltages[i], "Y:",y[i])
      

    elif measurement_mode == 'wait test': # CODE FOR wait_test()
    
        pos_1 = [0,1,0,1]
        pos_2 = [16,17,16,17]
        freq = 8000
        tc = 6 #9=30ms 8=10ms 7=3ms 6=1ms 5=300us 4=100us 3=30us
        interval =5E-4 #
        n_measurements = 100
        ClearSwitches(maintain_ground=False)
        R_array = wait_test(pos_1, pos_2, freq, interval, tc, n_measurements)
        R_all = []
        for i in range(0,5):
            r = wait_test(pos_1,pos_2,freq,interval,tc,n_measurements)
            R_all.append(r)
        R_mean = np.mean(R_all,axis=0)
        R_std = np.std(R_all,axis=0)
        R_mean_string = np.array2string(R_mean, precision=8, separator=',',suppress_small=True)
        R_std_string = np.array2string(R_std, precision=8, separator=',',suppress_small=True)
        #Currently just prints data, need to add in file saving
        print("y_tc"+str(tc)+" = np.array("+R_mean_string+")")
        print("yerr_tc"+str(tc)+" = np.array("+R_std_string+")")
        plt.xlabel(r'Delay time / s' , fontsize = 14, fontname = 'cmr10')
        plt.ylabel(r'Resistance / $\Omega $', fontsize = 12)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.grid(b=True, which='major', axis='both')
        time = np.arange(0.5, 50, 0.5)
        plt.errorbar(time,abs(R_mean), yerr=R_std, label='time constant: 1 ms, interval: 0.5 ms', marker='x', markersize=3,linestyle='none')
        plt.legend(loc='lower right')
        plt.show()
        
    else:
        print("no measurement mode selected")
    
    # ALWAYS CLEAR SWITCHES AND CLOSE RESOURCE MANAGER TO AVOID LOCKIN TIMEOUT
    # If the kernal had to be interrupted for any reason, call these manually
    lockin.write("IRNG 0")
    ClearSwitches(maintain_ground=True, n_el=n_el)
    print("Switches cleared, resource manager session closed")
    rm.close()
    
##################### DISCARDS ####################
    
# voltages_x = np.zeros((64,64))
# voltages_y = np.zeros((64,64))
# theta      = np.zeros((64,64))
# for i in range(0,len(x)):
#     voltages_x[positions[i][0]][positions[i][1]]=x[i]
#     voltages_y[positions[i][0]][positions[i][1]]=y[i]
    
# theta=np.arctan(voltages_y/voltages_x)*(360/(2*np.pi))


# plt.figure()
# im_x = plt.imshow(voltages_x,cmap='viridis',interpolation='nearest', origin='lower')
# plt.xlabel('Contact number' , fontsize = 12, fontname = 'cmr10')
# plt.ylabel('Contact number', fontsize = 12, fontname = 'cmr10')
# plt.title('X channel')
# plt.tick_params(axis='both', which='major', labelsize=12)
# plt.minorticks_on()
# plt.colorbar(im_x)
# plt.show()

# plt.figure()
# im_y = plt.imshow(voltages_y,cmap='viridis',interpolation='nearest', origin='lower')
# plt.xlabel('Contact number' , fontsize = 12, fontname = 'cmr10')
# plt.ylabel('Contact number', fontsize = 12, fontname = 'cmr10')
# plt.title('Y channel')
# plt.tick_params(axis='both', which='major', labelsize=12)
# plt.minorticks_on()
# plt.colorbar(im_y)
# plt.show()

# plt.figure()
# im_th = plt.imshow(theta,cmap='viridis',interpolation='nearest', origin='lower')
# plt.xlabel('Contact number' , fontsize = 12, fontname = 'cmr10')
# plt.ylabel('Contact number', fontsize = 12, fontname = 'cmr10')
# plt.title("Phase \u03B8 (\u00B0)")
# plt.tick_params(axis='both', which='major', labelsize=12)
# plt.minorticks_on()
# plt.colorbar(im_th)
# plt.show()

