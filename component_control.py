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

"""

import pyvisa
import numpy as np
from numpy import random
import time
import os
from datetime import datetime
from selection_algorithms import eit_scan_lines, voltage_meter, Standard, GetNextElectrodes, GetNextElectrodesESA
from get_next_prediction import reconstruction_plot, initialise_ex_volt_mat
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['cmr10']

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['cmr10']
mpl.rcParams['axes.unicode_minus'] = False
                         
#from lockin_test import TwoPointIndices 

#SET DEVICE IP AND PORT
lockin_ip = '169.254.147.1' #lock-in amplifier address
lockin_port = '1865'  #By default, the port is 1865 for the SR860.
lockin_lan_devicename = 'inst0' #By default, this is inst0. Check NI MAX

''' SWITCH USED IS ACCESSED VIA GPIB NOT LAN
switch_ip = '10.0.0.2' #default for CYTECH IF-6 module as used in VX256 is 10.0.0.2
switch_port = '23456' #default for CYTECH IF-6 module as used in VX256 is 23
'''
#SET DEVICE GPIB ADDRESS
#switchboard
switch_primary_address = '7'
#create devices (resources) address strings
switch_address = 'GPIB0::'+switch_primary_address+'::INSTR'
lockin_address = 'TCPIP::'+lockin_ip+'::'+lockin_lan_devicename+'::'+'INSTR'
rm = pyvisa.ResourceManager() #create resource manager using py-visa backend ('@py') leave empty for NI VIS
#print available devices (resources)
print(rm.list_resources())
switch = rm.open_resource(switch_address) #connect to devices
lockin = rm.open_resource(lockin_address)
switch.read_termination = '\n' #cytech manual says 'enter' so try \n, \r or combination of both
switch.write_termination = '\n'
#lockin.read_termination = '\f' #SR860 manual says \lf so \f seems to be equivalent in python)
lockin.write_termination = '\f'
print("switch", switch.session)
print("lockin", lockin.session)
#lockin.flush()

"""
------ FUNCTION DEFINITIONS -------
- SetMeasurementParameters(parameters)
- GetMeasurement(parameters=None, param_set=True)
- FlickSwitch(state, module, relay)
- MapSwitches(electrode, lockin_connection)
- ClearSwitches()
- eit_scan_lines(ne=16, dist=1)
- RunEIT(algorithm='Standard', no_electrodes=32, max_measurements=10000, measurement_electrodes = None, 
            print_status=True, voltage=2, freq=30, wait=60, tc=12, **algorithm_parameters)
- TimeStamp(fname, fmt='%Y-%m-%d-%H-%M-%S_{fname}')
- SaveEIT(positions, voltages, filename):
- TwoPointIndices(no_electrodes)
"""

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
    #switch.write(state_str+str(module)+" "+str(relay))
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

def eit_scan_lines(ne=16, dist=1):
    """
    TAKEN FROM pyeit.eit.utils.py
    Generates an excitation scan matrix of current and voltage electrode pairs.
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

def setup_electrodes(el_list, wait):
    """
    

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
    ClearSwitches()
    for i in range(0,4):
        module, relay = MapSwitches(electrode=el_list[i], lockin_connection=i)
        FlickSwitch(1, module, relay)
        #print("Module:", module,"relay:",relay)
    #print(switch.query('S'))
    time.sleep(wait)
    measurement = lockin.query('SNAPD?')
    measurement_array = np.fromstring(measurement, sep=',')
    Rin = 100e3
    V_in = 1

    I = V_in/Rin
    R = measurement_array[0]/I
    #print("R:",R)
    return measurement_array


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
    ClearSwitches()
    R_array = []  
    
    for j in range(0, n_measurements-1):
        ClearSwitches()
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
    '''
    print("Testing switchbox...")
    ClearSwitches()
    FlickSwitch('on', 0, 1)
    FlickSwitch('on', 1, 2)
    FlickSwitch('on', 2, 1)
    FlickSwitch('on', 3, 2)
    print("Switch status:", switch.query("S"))
    '''
    print("Initialising lock-in....")
    SetMeasurementParameters(["X","Y","THeta","FInt"])
    lockin.write("ISRC 1") #Set voltage unput to differential mode (A-B)
    lockin.write("SLVL " + str(voltage))
    lockin.write("FREQ " + str(freq))     # frequency
    lockin.write("OFLT " + str(tc)) # time constant 11 = 300ms, 12 = 1s
    lockin.write("PHAS 0") # set phase offset to 0
    #lockin.write("ASCL")   # autoscale
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
    #standard_measurement_electrodes = Standard(no_electrodes=no_electrodes, step=1,parser='fmmu')
    #print(standard_measurement_electrodes)    
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
                current_pairs_to_return = 2 #(REDUNDANT SEE EX_LINE_END IN get_next_preidcition) no. of current pairs to find voltages for an propose, 1-3 recommended
                voltages_to_return = 50 # no. of voltages per current pairs, 10-n_el//2 recommended
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
                                                             measurement=k, all_measurement_electrodes = measurement_electrodes, **algorithm_parameters)
            next_electrodes = next_electrodes.astype(np.int32)
            if keep_measuring == False:
                break
            #print(next_electrodes)
            start_clear = time.time()
            ClearSwitches()
            end_clear = time.time()
            clear_time = end_clear - start_clear
            flick_times.append(clear_time)
            #next_electrodes = np.random.randint(no_electrodes, size=(2,4))
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
                
                #if measurement==0:
                    #time.sleep(0.5*(1000/freq))
                    #time.sleep(0)
                time.sleep(wait * (1/freq)) # Wait to let lockin settle down - may not be nesceaary
                x=0
                y=0
                theta=0
                fint=0
                #if measurement == 0:
                    #time.sleep((1/freq)*wait)
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
                #print("i", i)
                #print('next electrodse[j]', next_electrodes[i])
                
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
    
    

def TimeStamp(fname, fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
    """
    Gets current time stamp for datafile saving.
    """
    return datetime.now().strftime(fmt).format(fname=fname)

def SaveEIT(positions, voltages, file):
    """
    Saves data of measurements generated from RunEIT.

    Parameters
    ----------
    positions : TYPE
        DESCRIPTION.
    voltages : TYPE
        DESCRIPTION.
    filename : TYPE
        DESCRIPTION.

    Returns
    -------
    filename : TYPE
        DESCRIPTION.

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
            
    data = [positions[:,0], positions[:,1], positions[:,2] ,positions[:,3], voltages]
    data = np.asarray(data).T
    np.savetxt(filename, data, fmt=['%i', '%i', '%i', '%i', '%e'], delimiter=",", header="sin+,sin-,v_high,v_low,voltage", comments="")
    print("File saved as", str(filename))
    return filename

def read_data_EIT(filename):
    positions = np.loadtxt(filename, delimiter=",", skiprows=1, usecols=[0,1,2,3])
    voltages = np.loadtxt(filename, delimiter=",", skiprows=1, usecols=4)

    return voltages, positions

def TwoPointIndices(no_electrodes):

    posns = np.zeros((no_electrodes, 4))

    for j in range(0, no_electrodes):
        posns[j,0] = j
        posns[j,1] = (j+1) % 32
        posns[j,2] = (j) % 32 
        posns[j,3] = (j+1) % 32

    return posns

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
                    print(R[:,l])
                for m in range(0, np.shape(positions)[0]): 
                    R_mean = np.mean(R[m,:])
                    R_std = np.std(R[m,:])
                    R_std_per = (R_std/R_mean) * 100
                    print("Positions:", positions[m,:])
                    print("R:", R_mean, "±", R_std_per, "%")
        
    return



if  __name__ == "__main__":
    
    #dist for current, step for voltages
    #voltages, positions, mesh, switch_times, lockin_times, y = RunEIT(no_electrodes=32, algorithm='Standard', freq=8000,tc=6,wait=160,dist=1, step=1, print_status=False)
    #voltages, positions, mesh, switch_times, lockin_times, y = RunEIT(no_electrodes=32, algorithm='ESA', freq=8000,tc=6,wait=160, max_measurements=10)
    #el_array = np.array([[1,2,1,2],[0,3,0,3]])
    el_array, ex_mat, volt_mat, ind = initialise_ex_volt_mat(current_mode='adj', volt_mode='adj', n_el=32, ex_spaced=False,ex_mat_length=None)
    voltages, positions, mesh, switch_times, lockin_times, y = RunEIT(no_electrodes=32, voltage=1, algorithm='Standard',
                   measurement_electrodes=el_array, freq=8000,tc=6,wait=160, print_status=False)
    arbit_scaling = 1e5
    #mesh_obj=None
    reconstruction_plot(positions, voltages*arbit_scaling, mesh_obj=mesh, start_pos='mid', n_el=32, p=0.5, lamb=0.02, n_pix=128)
    #filename = "grid_init-10-adj-adj-all-notspaced_8000Hz-tc6-wait160-ESA_p0-2_lamb0-1"
    #filename = "newgrid_adj-opp_8000Hz-tc6-wait160_p0-2_lamb0-1"
    #filename = "grid_init-10-opp-adj-all-spaced_8000Hz-tc6-wait240-ESA-Ipairs3-Vpairs30_p0-1_lamb0-05"
    #filename = "grid_init-10-opp-adj-all-spaced_30Hz-tc11-wait60-ESA-Ipairs3-Vpairs30_p0-2_lamb0-1"
    #filename = "newgrid_10-opp-adj-all-spaced_8000Hz-tc6-wait160-ESA-Ipairs2-Vpairs20_p0-5_lamb0-02"
    filename = "all_measurements_32_electrodes"
    #filename = "opp-adj-32-from-ESA-vanilla-check"
   # SaveEIT(positions, voltages, filename)
    
    
    
    # two_point = np.array([[0,3,0,3],[0,3,0,3],[0,3,0,3],[0,3,0,3],[0,3,0,3],
    #                       [1,2,1,2],[1,2,1,2],[1,2,1,2],[1,2,1,2],[1,2,1,2],
    #                       [8,11,8,11],[8,11,8,11],[8,11,8,11],[8,11,8,11],[8,11,8,11],
    #                       [15,18,15,18],[15,18,15,18],[15,18,15,18],[15,18,15,18],[15,18,15,18],
    #                       [16,17,16,17],[16,17,16,17],[16,17,16,17],[16,17,16,17,],[16,17,16,17]])
    # two_point2 = np.array([[0,3,0,3],[1,2,1,2],[0,3,1,2],[8,11,8,11],[10,13,10,13],[11,14,11,14],[15,18,15,18],[16,17,16,17],
    #                       [22,25,22,25],[24,27,24,27],[26,29,26,29]])
    # two_point3 = np.array([[0,3,0,3],[1,2,1,2],[0,3,1,2]])

    
    # four_point = np.array([[0,3,1,2],[0,3,1,2],[0,3,1,2],[0,3,1,2],[0,3,1,2],
    #                       [1,4,2,3], [1,4,2,3], [1,4,2,3], [1,4,2,3], [1,4,2,3],
    #                       [8,11,9,10],[8,11,9,10],[8,11,9,10],[8,11,9,10],[8,11,9,10],
    #                       [21,24,22,23],[21,24,22,23],[21,24,22,23],[21,24,22,23],[21,24,22,23],
    #                       [15,18,16,17],[15,18,16,17],[15,18,16,17],[15,18,16,17],[15,18,16,17]])
                          
    # two_point_adj = np.array([[0,1,0,1],[1,2,1,2],[8,9,8,9],[10,9,10,9]])
    #all_two_points = TwoPointIndices(32)
    # #all_two_points = np.flip(all_two_points,axis=0)
    # #two_points = np.array([[0,3,0,3],[8,11,8,11],[10,13,10,13],[11,14,11,14], [22,25,22,25], [24,27,24,27], [26,29,26,29]])
    # # two_points = np.array([[1,2,1,2],[1,2,1,2],[1,2,1,2],[1,2,1,2],[1,2,1,2],[1,2,1,2],[1,2,1,2],[1,2,1,2],[1,2,1,2],
    # #                        [15,16,15,16],[15,16,15,16],[15,16,15,16],[15,16,15,16],[15,16,15,16],[15,16,15,16],[15,16,15,16],
    # #                         [8,9,8,9],[8,9,8,9],[8,9,8,9],[8,9,8,9],[8,9,8,9],[8,9,8,9],[8,9,8,9],[8,9,8,9],[8,9,8,9],[8,9,8,9]])
    # two_points = np.array([[1,2,1,2],[1,2,1,2],[1,2,1,2],[1,2,1,2],[1,2,1,2],[1,2,1,2]])
   # noise_check_points = np.array([[0,1,0,1],[8,9,8,9],[22,23,22,23]])
    # #two_points = np.flip(two_points, axis=0)
    # #np.vstack((all_two_points,all_two_points))
    # freq_timeconstant_wait = [[8000, 10, 2*8000],[30,5,30*100]]
    # f_tc_input = freq_timeconstant_wait[0]
    #el_array = all_two_points
    Rin = 100e3
    V_in = 1
    I = V_in/Rin
    
    #voltages, positions, switch_times, lockin_times, y = RunEIT(no_electrodes=32, voltage=V_in, algorithm='Standard',
    #                                                            freq=8000,tc=6,wait=180, print_status=True,
    #                                                            step=1,dist=16)
    
    #SaveEIT(positions, voltages, "step1-dist16-eit-8khz-tc=5-wait=25ms")
    
    
    
    # V_all = []
    # for i in range(0,5):
    #     v, positions, mesh, switch_times, lockin_times, y = RunEIT(no_electrodes=32, voltage=V_in, algorithm='Standard',
    #               measurement_electrodes=el_array, freq=8000,tc=6,wait=160, print_status=True)
    #     V_all.append(v)
        
    # V_all = np.asarray(V_all)
    # voltages = np.mean(V_all,axis=0)
    # R_all = V_all/I
    # R_mean = np.mean(R_all,axis=0)
    # R_std = np.std(R_all,axis=0)
    # for i in range (0, len(R_mean)):
    #     print(positions[i]," R:",R_mean[i],"±",R_std[i]," V:",voltages[i])
    # # R_mean_string = np.array2string(R_mean, precision=8, separator=',',suppress_small=True)
    # # R_std_string = np.array2string(R_std, precision=8, separator=',',suppress_small=True)
    # # print(R_mean_string)
    # # print(R_std_string)
    # #filename = "step1-dist16-eit-8khz-tc=5-wait=25ms"
    # filename = "new-grid-two-point-resistor-check"
    # SaveEIT(positions, voltages, filename)
    # I = V_in/Rin
    # R = voltages/I
    # for i in range(0,len(voltages)):
    #     print(positions[i],"R:",R[i], "V:",voltages[i], "Y:",y[i])
    '''
    freqs = [8000]
    tcs = [6] #10 is 0.1s
    waits = [160]
    noise_check(all_two_points,freqs,tcs,waits,reps=5)
    '''
    
    
    #BELOW IS CODE FOR wait_test()
    
    
    

    
    
    # pos_1 = noise_check_points[1]
    # pos_2 = noise_check_points[2]
    
    # freq = 8000
    # tc = 6 #9=30ms 8=10ms 7=3ms 6=1ms 5=300us 4=100us 3=30us
    # interval =5E-4 #
    # n_measurements = 100
    
    # R_array = wait_test(pos_1, pos_2, freq, interval, tc, n_measurements)
    # R_all = []
    
    # for i in range(0,10):
    #     r = wait_test(pos_1,pos_2,freq,interval,tc,n_measurements)
    #     R_all.append(r)
    # R_mean = np.mean(R_all,axis=0)
    # R_std = np.std(R_all,axis=0)
    # R_mean_string = np.array2string(R_mean, precision=8, separator=',',suppress_small=True)
    # R_std_string = np.array2string(R_std, precision=8, separator=',',suppress_small=True)

    # print("y_tc"+str(tc)+" = np.array("+R_mean_string+")")
    # print("yerr_tc"+str(tc)+" = np.array("+R_std_string+")")
    
    ClearSwitches()
    rm.close()

    

    
    
    

# #initialise devices
# #open all switches
# switch.write('C') #C = Clear = open all relays or turn OFF all relays
# switch_status = switch.query('S') #S = Status which can be used on individual switch points, modules, or the entire system. Replies with string eg 0000000 showing 8 closed switches
# print(switch_status)
# switch.write('L3 5')
# print(switch.query('S'))

# #reset lock-in
# print("resetting lock-in")
# #check timebase is internal
# tb_status = lockin.query('TBSTAT?') #Query the current 10 MHz timebase ext (0) or int (1)
# print("Timebase status:", tb_status)

# #set reference phase
# PHASE_INIT = 0
# lockin.write('PHAS '+str(PHASE_INIT)) #Set the reference phase to PHASE_INIT
# phase = lockin.query('PHAS?') #Returns the reference phase in degrees
# print("Reference phase:", phase)

# #set frequency
# FREQ_INIT = 1e4
# lockin.write('FREQINT '+ str(FREQ_INIT)) #Set the internal frequency to FREQ_INIT
# freq = lockin.query('FREQINT?') #Returns the internal frequency in Hz
# print("Frequency: ", freq)

# #set sine out voltage
# VOUT_INIT = 0.5
# lockin.write('SLVL '+str(VOUT_INIT)) #Set the sine out amplitude to VOUT_INT in Volts The amplitude may be programmed from 1 nV to 2.0V
# vout = lockin.query('SLVL?') #Returns the sine out amplitude in Volts
# print("Sine out amplitude: ", vout)


# #Assign parameters to data channels. Lock-in is capable or reading 4 data points simultaneously. 
# lockin.write('CDSP DAT1 R')         #set channel 1 to R
# lockin.write('CDSP DAT2 THetha')    #set channel 2 to theta
# lockin.write('CDSP DAT3 SAMp')        #set channel 3 to sine out amplitude
# lockin.write('CDSP DAT4 FInt')        #set channel 4 to internal reference frequency

# #Auto adjust range and scaling of measurements
# lockin.write("ARNG") #auto range 
# lockin.write("ASCL") #auto scale


# params = ["X", "Y", "OUT1", "OUT2"]
# SetMeasurementParameters(params)
# start_false = time.time()
# data = GetMeasurement(param_set=False)
# end_false=time.time()
# time_false = -(start_false-end_false)
# print("Time to get measurement without setting params:", time_false)

# print(data)
# start_true = time.time()
# data = GetMeasurement()
# end_true=time.time()
# time_true = end_true - start_true
# print("Time to get measurement with setting params:", time_true)
# print(data)

# #INSERT CODE HERE 