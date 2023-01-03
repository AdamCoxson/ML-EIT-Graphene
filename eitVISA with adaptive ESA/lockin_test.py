"""
Created on Tue Oct 27 10:28:08 2020
Authors: Adam Coxson and Frederik Brookebarnes, MPhys Undergrads, The University of Manchester
Project: Automated Electrical Impedance Tomography of Graphene
Module:  lockin_test
Dependancies: pyVisa package

This script is used to test the lockin amplifier and switchbox interaction with
pyVisa.
"""

import pyvisa
import numpy as np
from numpy import random
import time
from datetime import datetime
import os


# from meshing import mesh
#from selection_algorithms import *
#from component_control import SetMeasurementParameters, MapSwitches, FlickSwitch, ClearSwitches, GetMeasurement, TimeStamp,open_next_switches



#SET DEVICE IP AND PORT
lockin_ip = '169.254.147.1' #lock-in amplifier address
lockin_port = '1865'  #By default, the port is 1865 for the SR860.
lockin_lan_devicename = 'inst0' #By default, this is inst0. Check NI MAX

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

#SetMeasurementParameters("X, Y, THeta, FInt")

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

def GetMeasurement(freq, repeats=5, n_periods=5):
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
        Array of floats corresponding to measurement values in Volts, Hz or Degrees. Ordered in same order as specified in parameters.
    '''
    x = np.zeros(repeats)
    y = np.zeros(repeats)
    theta = np.zeros(repeats)
   
    for i in range(0,repeats):
        time.sleep(1/freq * n_periods)
        measurement = lockin.query('SNAPD?')
        measurement_array = np.fromstring(measurement, sep=',')
        x[i] = measurement_array[0]
        y[i] = measurement_array[1]
        theta[i] = measurement_array[2]
        
    X = np.mean(x)
    X_std = np.std(x)
    Y = np.mean(y)
    Y_std = np.std(y)
    THETA = np.mean(theta)
    THETA_std = np.std(theta)
    
    return X, X_std, Y, Y_std, THETA, THETA_std

def query_lockin_settings():
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

def open_next_switches(new_electrodes, old_electrodes):
    
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

def lockin_troubleshoot(freq=8000, tc=6, wait=500, pos_1=[0,3,1,2], pos_2=[14,17,15,16]):
    I = 1e-4
    lockin.write("FREQ " + str(freq))
    lockin.write("OFLT " + str(tc))
    
    ClearSwitches(maintain_ground=True, n_el=64)
    open_next_switches(pos_1, [0,0,0,0])
    time.sleep(0.1)
    open_next_switches(pos_2, pos_1)
    time.sleep(wait*(1/freq))
    x, x_err, y, y_err, theta, theta_std = GetMeasurement(freq, repeats=10, n_periods=10)
    #print("Electrodes",pos_1,"to",pos_2, "X:",x,"±",x_err,",",(100*x_err)/x,"%")
    freq=int(lockin.query("FREQ?"))
    tc=int(lockin.query("OFLT?"))
    print("Freq:",freq,"tc:",tc,"wait:",wait )
    print("X:",x,"±",x_err,"R:",x/I,"±",x_err/I," ",(100*x_err)/x,"%")
    return x, x_err, y, y_err, theta, theta_std

def TimeStamp(fname, fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
    """
    Gets current time stamp for datafile saving.
    """
    return datetime.now().strftime(fmt).format(fname=fname)

# def Save_freq(freq_tc_wait_list, x, x_err, y, y_err, theta, theta_err, file='default'):

#     print("Saving...")
#     file = TimeStamp(file+".csv")
#     filename = os.getcwd() + "\\data\\" + file
#     filename = file
#     data = [freq_tc_wait_list[:,0], freq_tc_wait_list[:,1], freq_tc_wait_list[:,2], x, x_err, y, y_err, theta, theta_err]
#     data = np.asarray(data).T
#     np.savetxt(filename, data, fmt=['%e','%e','%e','%e','%e','%e','%e','%e','%e'], delimiter=","
#                ,header="freq,tc,wait,x,x_err,y,y_err,theta,theta_err", comments="pos_1"+str(pos_1)+"pos_2")
#     print("File saved as", str(filename))
#     return filename

def frequency_sweep(freq_tc_wait_list, pos_1, pos_2, I=1e-4, file='default'):
    SetMeasurementParameters(["X","Y","THeta","FInt"])
    
    x = np.zeros(len(freq_tc_wait_list))
    x_err = np.zeros(len(freq_tc_wait_list))
    y = np.zeros(len(freq_tc_wait_list))
    y_err = np.zeros(len(freq_tc_wait_list))
    theta = np.zeros(len(freq_tc_wait_list))
    theta_err = np.zeros(len(freq_tc_wait_list))
    freq = np.zeros(len(freq_tc_wait_list))
    tc = np.zeros(len(freq_tc_wait_list))
    tc_val = np.zeros(len(freq_tc_wait_list))
    wait = np.zeros(len(freq_tc_wait_list))
    tc_map = [1.0e-6,3.0e-6, 1.0e-5,3.0e-5, 1.0e-4,3.0e-4, 1.0e-3,3.0e-3, 1.0e-2,3.0e-2, 1.0e-1,3.0e-1, 1.0, 3.0]
    for i in range(0,len(freq_tc_wait_list)):
        freq[i] = freq_tc_wait_list[i][0]
        tc[i] = freq_tc_wait_list[i][1]
        wait[i] = freq_tc_wait_list[i][2]
    
    for i in range(0,len(freq_tc_wait_list)):
        x[i], x_err[i], y[i], y_err[i],theta[i],theta_err[i] = lockin_troubleshoot(freq=freq[i], tc=tc[i], wait=wait[i],
                                                                                   pos_1=pos_1, pos_2=pos_2)
        tc_val[i] = tc_map[int(tc[i])]
        
    print("Saving...")
    file = TimeStamp(file+".csv")
    filename = os.getcwd() + "\\data\\" + file
    filename = file
    data = [freq, tc_val, wait, x, x_err, y, y_err, theta, theta_err]
    data = np.asarray(data).T
    comment_str_1 = "pos_1,"+str(pos_1[0])+" "+str(pos_1[1])+" "+str(pos_1[2])+" "+str(pos_1[3])+",pos_2,"+str(pos_2[0])+" "+str(pos_2[1])+" "+str(pos_2[2])+" "+str(pos_2[3])
    comment_str_2 = ",Current: "+str(I)
    np.savetxt(filename, data, fmt=['%e','%e','%e','%e','%e','%e','%e','%e','%e'], delimiter=","
               ,header="freq (Hz),tc (s),wait (periods),x,x_err,y,y_err,theta,theta_err",
               #comments = "pos_1,"+str(pos_1)+",pos_2,"+str(pos_2)+"\n")
               comments=comment_str_1+comment_str_2+"\n")
    print("File saved as", str(filename))
    return x, x_err, y, y_err, theta, theta_err

def frequency_sweep_two_point(freq_tc_wait_list, I=1e-4, file='default'):
    SetMeasurementParameters(["X","Y","THeta","FInt"])
    x = np.zeros(len(freq_tc_wait_list))
    x_err = np.zeros(len(freq_tc_wait_list))
    y = np.zeros(len(freq_tc_wait_list))
    y_err = np.zeros(len(freq_tc_wait_list))
    theta = np.zeros(len(freq_tc_wait_list))
    theta_err = np.zeros(len(freq_tc_wait_list))
    freq = np.zeros(len(freq_tc_wait_list))
    tc = np.zeros(len(freq_tc_wait_list))
    tc_val = np.zeros(len(freq_tc_wait_list))
    wait = np.zeros(len(freq_tc_wait_list))
    tc_map = [1.0e-6,3.0e-6, 1.0e-5,3.0e-5, 1.0e-4,3.0e-4, 1.0e-3,3.0e-3, 1.0e-2,3.0e-2, 1.0e-1,3.0e-1, 1.0, 3.0]
    for i in range(0,len(freq_tc_wait_list)):
        freq[i] = freq_tc_wait_list[i][0]
        tc[i] = freq_tc_wait_list[i][1]
        wait[i] = freq_tc_wait_list[i][2]
    
    
    for i in range(0,len(freq_tc_wait_list)):
        lockin.write("FREQ " + str(freq[i]))     # frequency
        lockin.write("OFLT " + str(tc[i])) # time constant 11 = 300ms, 12 = 1s
        time.sleep(0.2)
        time.sleep((1/freq[i])*wait[i])
        x[i], x_err[i], y[i], y_err[i],theta[i],theta_err[i] = GetMeasurement(freq[i], repeats=10, n_periods=20)
        tc_val[i] = tc_map[int(tc[i])]
        
    print("Saving...")
    file = TimeStamp(file+".csv")
    filename = os.getcwd() + "\\data\\" + file
    filename = file
    data = [freq, tc_val, wait, x, x_err, y, y_err, theta, theta_err]
    data = np.asarray(data).T
    comment_str_2 = "OFSL=0, 6 dB. Current: "+str(I)
    np.savetxt(filename, data, fmt=['%e','%e','%e','%e','%e','%e','%e','%e','%e'], delimiter=","
               ,header="freq (Hz),tc (s),wait (periods),x,x_err,y,y_err,theta,theta_err",
               comments=comment_str_2+"\n")
    print("File saved as", str(filename))
    return x, x_err, y, y_err, theta, theta_err
    
    
def query_lockin_settings():
    settings = ['TBMODE?','TBSTAT?','PHAS?','FREQ?','FREQINT?','FREQEXT?','FREQDET?','HARM?',
    'HARMDUAL?','BLADESLOTS?','BLADEPHASE?','SLVL?','SOFF?','REFM?','RSRC?','RTRG?',
    'REFZ?','IVMD?','ISRC?','ICPL?','IGND?','IRNG?','ICUR?',
    'ILVL?','SCAL?','OFLT?','OFSL?','SYNC?','ADVFILT?','ENBW?']
    
    for i in range(0,len(settings)):
        string = lockin.query(settings[i])
        print(settings[i],",",string)
    
    return None

def TwoPointIndices(no_electrodes):

    posns = np.zeros((no_electrodes, 4))

    for j in range(0, no_electrodes):
        posns[j,0] = j
        posns[j,1] = (j+1) % 32
        posns[j,2] = (j) % 32 
        posns[j,3] = (j+1) % 32

    return posns
def TwoPointIndicesSkip(no_electrodes):

    posns = np.zeros((no_electrodes, 4))

    for j in range(0, no_electrodes):
        posns[j,0] = j
        posns[j,1] = (j+3) % 32
        posns[j,2] = (j) % 32 
        posns[j,3] = (j+3) % 32

    print("infunc posns", posns)
    return posns

def Measure(posns, voltage=1, freq=30, tc=11, wait=60):

    
    lockin.write("FREQ " + str(freq)) #frequency
    lockin.write("OFLT " + str(tc)) #time constant 11 = 300ms
    
    x = []
    theta = []
    y = []
    fint = []
    for i in range(0, len(posns)):
        ClearSwitches()
        #print("switches cleared")
        for j in range(0, 4):
            electrode = int(posns[i][j])
            lockin_connection= j
            #print("electrode", electrode)
            #print("lockin connect", lockin_connection)
            module, relay = MapSwitches(electrode=electrode, lockin_connection=lockin_connection)
            #print("module", module)
            #print("relay", relay)
            FlickSwitch(state=1, module=module, relay=relay)
            #print("switch status", switch.query('S'))
        
        #print("Waiting...")
       # time.sleep(wait * (1/freq))
        print("Taking measurement", i)
        time.sleep(wait)
        x_i, theta_i, y_i, fint_i  = GetMeasurement(param_set=False)
        #print("x")
        print("x_i", x_i)
        #print("Measurement "+str(i)+" done.")
        #print("data", data)
        x.append(x_i)
        theta.append(theta_i)
        y.append(y_i)
        fint.append(fint_i)
        

    return x, theta, y, fint
def FreqSweep(posns, freqs, tcs, voltage, wait=100):


    x = np.zeros((len(posns), len(freqs)))
    theta = np.zeros((len(posns), len(freqs)))
    y = np.zeros((len(posns), len(freqs)))
    fint = np.zeros((len(posns), len(freqs)))

    for i in range(0, len(freqs)):
            
        x[:,i], theta[:,i], y[:,i], fint[:,i] = Measure(posns = posns,
                                                        voltage=voltage, freq=freqs[i], tc=tcs[i], wait=wait[i])
        
    return x, theta, y, fint


def RunSweep(repeats=25):
    """
    For both 8kHz and 30Hz we found that the best compromise between minimising the wait time and minimising the
    error was a time constant ~3 times the driving period and a waiting time ~10 times the time constant.

    The tc variable is the time constant command for the lock-in:
    tc=0 is 1us, 1=3us, 2=10us, 3=30us, 4=100us,...11=300ms, 12 = 1s 
    Returns
    -------
    None.

    """

    #frequencies = np.array( [5, 10,20,30,40,50,70,1e2,1.5e2,2e2,3e2,6e2,1e3,2e3,3e3,1e4,3e4,1e5,3e5])
    #tcs = np.array(         [13,13,12,12,11,11,11,11, 10,   10, 10, 9,  9,  8,  8,  7,  6,  5,  4])

    # frequencies = np.array([30, 50, 70, 90,
    # 100, 150, 250,
    # 3e2, 6e2, 9e2,
    # 1e3, 1.5e3, 2e3,
    # 3e3, 5e3, 6e3, 7e3, 8e3, 8.5e3, 9e3,
    # 1e4, 1.2e4, 1.5e4, 2e4,
    # 3e4, 6e4, 9e4,
    # 1e5, 1.5e5, 2e5,
    # 3e5])
    
    # tcs =   np.array([13,13,13,13,
    # 12,12,12,
    # 11,11,11,
    # 10,10,10,
    # 9,9,9,9,9,9,9,  
    # 8,8,8,8,
    # 7,7,7,
    # 6,6,6,
    # 5,5,5, 
    # 4])
    # tcs =   np.array([10,10,10,10, # 100ms
    # 9,9,9,                         # 30ms
    # 8,8,8,                         # 10ms
    # 7,7,7,                         
    # 6,6,6,6,6,6,6,                 # 1ms
    # 5,5,5,5, 
    # 4,4,4,                         # 100us
    # 3,3,3,                         # 
    # 3])                            # 30us
    
    # factor =1
    # #tc=11 is 300ms
    # tcs = np.array([3,4,5,6,7,8,9])
    # frequencies = np.array([3.333e4,1e4,3.333e3,1e3,3.333e2,1e2,3.333e1])  * factor   
    # f_factor =1
    # w_factor = 20
    # #tc=11 is 300ms
    # tcs = np.array([3,4,4,5,5,5,6,6,7,7,7,8,8,9,9,9,10,10,11,11,11])
    # frequencies = np.array([3.3333e5,2e5,1e5,8e4,5e4,3.3333e4,2e4,1e4,8e3,5e3,3.3333e3,2e3,1e3,8e2,5e2,3.3333e2,2e2,1e2,8e1,5e1,3.3333e1])*f_factor   
    # wait = (1/frequencies)*w_factor*f_factor# wait time ~ 30 times the period
    
    f_factor = 10
    w_factor = 16
    tcs = np.array([2,
                    3,3,3,
                    4,4,4,
                    5,5,5,
                    6,6,6,
                    7,7,7,
                    8,8,8,
                    9,9,9,
                    10,10,10])
    frequencies = np.array([3.333e4,
                            1e4, 1.3e4, 2e4,
                            3.333e3, 5e3, 8e3,
                            1e3, 1.3e4, 2e3,
                            3.333e2, 5e2, 8e2,
                            1e2, 1.3e4, 2e2,
                            3.333e1, 5e1, 8e1,
                            1e1, 1.3e4, 2e1,
                            3.333, 5, 8])*f_factor   
    wait = (1/frequencies)*w_factor* f_factor# wait time ~ 30 times the period
    voltage = 1
    #wait[13:]=wait[13] # Set minimum bound of wait time to 10ms

    #tcs = tcs
    #frequencies = np.array([1e5,3e5])
    #tcs = np.array([5,4])
    
    #wait = 100 #time to wait between measurements divided by (1/f) of driving frequency ie no. of periods
    SetMeasurementParameters(["X","THeta","Y","FInt"])
    lockin.write("SLVL " + str(voltage))
    lockin.write("ISRC 1") #set voltage input to differntial mode (A-B) 
    lockin.write("OFSL 0")
    lockin.write("PHAS 0") 
    lockin.write("SCAL 21")
    lockin.write("IRNG 4")
    #positions = SideInidices(32)
    #positions = np.asarray([[0,1,3,2],[7,8,10,9]])
    positions = np.asarray([[0,3,1,2],[16,19,17,18]])


    x_array = np.zeros((np.shape(positions)[0], np.size(frequencies), repeats))
    theta_array = np.zeros((np.shape(positions)[0], np.size(frequencies), repeats))
    y_array = np.zeros((np.shape(positions)[0], np.size(frequencies), repeats))
    fint_array = np.zeros((np.shape(positions)[0], np.size(frequencies), repeats))

    for i in range(0,repeats):
        x_array[:,:,i], theta_array[:,:,i], y_array[:,:,i], fint_array[:,:,i] = FreqSweep(posns=positions, freqs=frequencies, tcs=tcs, voltage=voltage, wait=wait)

        
    
    x = np.mean(x_array, axis=2)
    y = np.mean(y_array, axis=2)
    theta = np.mean(theta_array, axis=2)
    fint = np.mean(fint_array, axis=2)
    x_std = np.std(x_array, axis=2)
    y_std = np.std(y_array, axis=2)
    theta_std = np.std(theta_array, axis=2)
    calc_theta_array = x_array/((x_array**2+y_array**2)**0.5)
    calc_theta = np.mean(calc_theta_array,axis=2)
    calc_theta_std = np.std(calc_theta_array,axis=2)
    print("x_array:", x_array)
    print("x", x)

    
    filename = "freq-sweep"+"-f_factor="+str(f_factor)+"-w_factor="+str(w_factor)
    for i in range(0, len(frequencies)):
        filename_csv="freq_sweep-"+str(frequencies[i])+"Hz"
        filename_csv = TimeStamp(filename+".csv")
        data = [x[:,i], x_std[:,i], theta[:,i], theta_std[:,i],  y[:,i], y_std[:,i], fint[:,i], positions[:,0], positions[:,1], positions[:,2], positions[:,3]]
        data = np.asarray(data).T
        np.savetxt(filename_csv, data, fmt=['%e', '%e', '%e', '%e', '%e', '%e', '%e', '%i', '%i', '%i', '%i'], delimiter=",", header="[x,x_std, theta, theta_std, y, y_std,fint,sin+,sin-,v+,v-]", comments=(str(frequencies[i])+"Hz wait="+str(wait)+"periods"+" tc="+str(tcs[i])))
    
    filename_npz = TimeStamp(filename+".npz")
    np.savez(filename_npz, x=x, x_std=x_std, theta=theta,calc_theta=calc_theta,calc_theta_std=calc_theta_std, theta_std=theta_std, y=y, y_std=y_std, fint=fint, posns=positions, tcs=tcs, freqs=frequencies)
    freq_sweep_data = np.load(filename_npz)
    print(filename_npz)
    print(freq_sweep_data['x'])
    #print("calc theta:\n",calc_theta," +/- ", calc_theta_std)

    return 0

def RunTwoPointMeasurement(posns, voltage, freq, tc, wait,filename="two-point"):

    #x = np.zeros(len(posns))
    #theta = np.zeros(len(posns))
    #y = np.zeros(len(posns))
    #fint = np.zeros(len(posns))
    x, theta, y, fint = Measure(posns = posns, voltage=voltage, freq=freq, tc=tc, wait=wait)

    filename_csv = TimeStamp(filename+".csv")
    data = [x, theta,  y, fint, posns[:,0], posns[:,1], posns[:,2], posns[:,3]]
    data = np.asarray(data).T
    np.savetxt(filename_csv, data, fmt=['%e', '%e', '%e', '%e', '%i', '%i', '%i', '%i'], delimiter=",", header="[x,theta, y,fint,sin+,sin-,v+,v-]", comments=str(freq)+"Hz wait="+str(wait)+"periods"+" tc="+str(tc))
    filename_npz = TimeStamp(filename+".npz")
    np.savez(filename_npz, x=x, theta=theta, y=y, fint=fint, posns=posns, tcs=tc, freqs=freq)
    two_point_data = np.load(filename_npz)
    print(filename_npz)
    #print(two_point_data['x'])
    return filename_npz

#positions = TwoPointIndices(32)
#print("main positions ", positions)
n_el=64
freq=330
voltage = 1
tc=10
SetMeasurementParameters(["X","Y","THeta","FInt"])
lockin.write("ISRC 1") # Set voltage unput to differential mode (A-B)
lockin.write("SLVL " + str(voltage))
lockin.write("FREQ " + str(freq))     # frequency
lockin.write("OFLT " + str(tc)) # time constant 11 = 300ms, 12 = 1s
lockin.write("PHAS 0") # set phase offset to 0
lockin.write("OFSL 2")
#lockin.write("SCAL 18")
lockin.write('SCAL 9')
lockin.write("IRNG 4")
time.sleep(0.5)

#ClearSwitches(maintain_ground=True, n_el=n_el)

pos_1 = [0,3,1,2]
pos_2=[4,5,4,5]
# freq_tc_wait_list = [[30,13,2000],[130, 11, 3000],[330, 11, 3000],[3000, 9 , 5000],[5000, 9 , 6000],
#                      [7330,9,8000],[9000,8, 8000],[10000,8, 8000],[13000,8, 10000]
#                      [12000,8, 10000],[14000, 8, 1.4e4],[20000,8,2e4],[3e4,7,3e4],
#                      [4e4,7,4e4],[5e4,7,5e4]]
# 18 dB slope
# freq_tc_wait_list = [[30,13,2000],[45,13,2000],[70,12,2000],[110, 12, 2000],[160,12,3000],[240,11,3000],[330, 11, 3000],[480,11,3000],[720,11,4000],
#                      [1000,10,4000],[1400,10,4000],[2000,10,5000],[3000, 9 , 5000],[4000,9,6000],[5000, 9 , 6000],[6000,9,7000],
#                      [7330,9,8000],[9000,8, 9000],[10000,8, 1e4],[1.2e4,8,1.2e4],[1.4e4,8,1.4e4],[1.7e4,8,1.7e4],[2e4,8,2e4],[2.5e4,8,2.5e4],
#                      [3e4,7, 3e4],[3.5e4,7,3.5e4],[4e4,7,4e4],[4.5e4,7,4.5e4],[5e4,7,5e4],[5.5e4,7,5.5e4],
#                        [6e4,7,6e4],[7.033e4,7,7.033e4],[8.033e4,6,8.033e4],[9.033e4,6,9.033e4],
#                        [1.0033e5,6, 1.0033e5],[2.0033e5,6, 2.0033e5],[4.0033e5,5, 4.0033e5], [5.00e5,5, 5.00e5]]
#6 dB slope
# freq_tc_wait_list = [[30,11,2000],[45,11,2000],[70,10,2000],[110, 10, 2000],[160,10,3000],[240,9,3000],[330, 9, 3000],[480,9,3000],[720,9,4000],
#                      [1000,8,4000],[1400,8,4000],[2000,8,5000],[3000, 7 , 5000],[4000,7,6000],[5000, 7 , 6000],[6000,7,7000],
#                      [7330,7,8000],[9000,6, 9000],[10000,6, 1e4],[1.2e4,6,1.2e4],[1.4e4,6,1.4e4],[1.7e4,6,1.7e4],[2e4,6,2e4],[2.5e4,6,2.5e4],
#                      [3e4,5, 3e4],[3.5e4,5,3.5e4],[4e4,5,4e4],[4.5e4,5,4.5e4],[5e4,5,5e4],[5.5e4,5,5.5e4],
#                        [6e4,5,6e4],[7.033e4,5,7.033e4],[8.033e4,4,8.033e4],[9.033e4,4,9.033e4],
#                        [1.0033e5,4, 1.0033e5],[2.0033e5,4, 2.0033e5],[4.0033e5,3, 4.0033e5], [5.00e5,3, 5.00e5]]

freq_tc_wait_list = [[10000,8, 1e4],[1.2e4,8,1.2e4],[1.4e4,8,1.4e4],[1.7e4,8,1.7e4],[2e4,8,2e4],[2.5e4,8,2.5e4],
                      [3e4,7, 3e4],[3.5e4,7,3.5e4],[4e4,7,4e4],[4.5e4,7,4.5e4],[5e4,7,5e4],[5.5e4,7,5.5e4],
                        [6e4,7,6e4],[7.033e4,7,7.033e4],[8.033e4,6,8.033e4],[9.033e4,6,9.033e4],
                        [1.0e5,6, 1.0e5],[1.2e5,6,1.2e5],[1.4e5,6,1.4e5],[1.7e5,6,1.7e5],[2e5,6,2e5],[2.5e5,6,2.5e5],
                      [3e5,6, 3e5],[3.5e5,6,3.5e5],[4e5,6,4e5],[4.5e5,6,4.5e5],[5e5,6,5e5]]
query_lockin_settings()
#freq_tc_wait_list = [[330, 11, 3000],[3e5, 5, 3e5],[5e5,5,5e5]] # for debug
#x, x_err, y, y_err, theta, theta_err = frequency_sweep_two_point(freq_tc_wait_list, file='freq_lockin_direct_6dB-slope')
x, x_err, y, y_err, theta, theta_err = frequency_sweep(freq_tc_wait_list, pos_1, pos_2, file='freq_two-point_second-half')
#query_lockin_settings()
print("Switch status:", switch.query("S"))
#ClearSwitches(maintain_ground=True, n_el=n_el)
#lockin_troubleshoot(freq=4e4, tc=7, wait=4e4, pos_1=[0,3,1,2], pos_2=[4,5,4,5])
#lockin.write("IRNG 0")
rm.close()
# print("Debug starting ...\n")

# el_array=[]
# for i in range(0,64):el_array.append([i,i+3,i+1,i+2])
# el_array[61:]=[[61,0,62,63],[62,1,63,0],[63,2,0,1]]

# for j in range(0,len(el_array)):

#     lockin_troubleshoot(freq=330, tc=10, wait=1000, pos_1=[40,48,1,2], pos_2=el_array[j])

# V =1
# Rin = 10e3
# #x, theta, y, fint = Measure(np.asarray([[0,1,3,2],[7,8,10,9]]), voltage=1, freq=8000, tc=6, wait=(1/8000)*30)
# RunSweep()
# print("Switch status:", switch.query("S"))
# ClearSwitches()

# rm.close()
"""
#filename_npz = "2020-11-26-14-17-39_two-point.npz"
filename_npz = RunTwoPointMeasurement(voltage=V, posns=positions, freq=1000, tc=9, wait=100,filename="twopoint")

two_point_data = np.load(filename_npz)
print(filename_npz)
print(two_point_data['x'])
x = two_point_data['x']
theta = two_point_data['theta']

print('x', x)
I = V/Rin
R = x / I

print("R", R)
for i in range(0, len(R)):
    print("n="+str(i)+" R="+str(R[i]))
    


for i in range(0, len(R)):
    print("electrode "+str(i)+" R = "+str(R[i])+"ohm")    


v_out = 2 #2V input voltage
shunt_resistor = 100e3
frequency = 100 # 30Hz frequency
tc = 11 #time constant 13->3s, 12->1s, 11->300ms, 10->100ms
positions = SideInidices(32)
results = Measure(positions, voltage=v_out, freq=frequency, tc=tc)
print(results)
voltages = np.asarray(results[0])
print("voltages")
print(voltages)
noise = np.asarray(results[2])
voltages_percantege_err = (noise/voltages) *100

current = v_out / shunt_resistor
resistances = voltages / current
print("Resistances")
for i in range(0, len(resistances)):
    print(str(resistances[i])+"+/-"+str(voltages_percantege_err[i])+"%")

print(switch.query('S'))
ClearSwitches()
print(switch.query('S'))
posns = np.array([0,3,1,2])
for j in range(0, 4):

    module, relay = MapSwitches(electrode = posns[j], lockin_connection=j)
    FlickSwitch("on", module, relay)
print(switch.query('S'))
"""

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