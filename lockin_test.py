import pyvisa
import numpy as np
from numpy import random
import time
from datetime import datetime
#import cupy as cp
# from get_next_prediction import adaptive_ESA_single_interation as adaptive_ESA
# from get_next_prediction import initialise_ex_volt_mat
# from meshing import mesh
from selection_algorithms import *
from component_control import SetMeasurementParameters, MapSwitches, FlickSwitch, ClearSwitches, GetMeasurement, TimeStamp



#SET DEVICE IP AND PORT
#lock-in amplifier
lockin_ip = '169.254.147.1'
lockin_port = '1865'  #By default, the port is 1865 for the SR860.
lockin_lan_devicename = 'inst0' #By default, this is inst0. Check NI MAX
#switchboard
'''
SWITCH USED IS ACCESSED VIA GPIB NOT LAN
switch_ip = '10.0.0.2' #default for CYTECH IF-6 module as used in VX256 is 10.0.0.2
switch_port = '23456' #default for CYTECH IF-6 module as used in VX256 is 23
'''
#SET DEVICE GPIB ADDRESS
#switchboard
switch_primary_address = '7'
#create devices (resources) address strings
switch_address = 'GPIB0::'+switch_primary_address+'::INSTR'
lockin_address = 'TCPIP::'+lockin_ip+'::'+lockin_lan_devicename+'::'+'INSTR'
#create resource manager using py-visa backend ('@py') leave empty for NI VIS
rm = pyvisa.ResourceManager()
#print available devices (resources)
print("lockin_test")
print(rm.list_resources())
#connect to devices
print(switch_address)
switch = rm.open_resource(switch_address)
print(switch.session)
print(lockin_address)
lockin = rm.open_resource(lockin_address)
#set termination characters
#switch
switch.read_termination = '\n' #cytech manual says 'enter' so try \n, \r or combination of both
switch.write_termination = '\n'
#lockin
#lockin.read_termination = '\f' #SR860 manual says \lf so \f seems to be equivalent in python)
lockin.write_termination = '\f'

SetMeasurementParameters("X, THeta, Y, FInt")

def Loop(freq, tc):
    lockin.write("FREQ" + str(freq))
    lockin.write("OFLT" + str(tc))

    x, theta, y, fint = GetMeasurement(param_set=False)

    return x, theta, y, fint

def Test(freqs, tc):

    x = []
    theta = []
    y = []
    fint = []
    for i in range(0, len(freqs)):

        data = Loop(freqs[i], tc)
        x.append(data[0])
        theta.append(data[1])
        y.append(data[2])
        fint.append(data[3])
    return x, theta, y, fint

def SideInidices(no_electrodes):

    posns = np.zeros((no_electrodes,4))

    for j in range(0, no_electrodes):
        posns[j,0] = j
        posns[j,1] = (j+3) % 32
        posns[j,2] = (j+1) % 32 
        posns[j,3] = (j+2) % 32

    return posns

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
'''
def TwoPointMeasurement(posns, tcs, voltage, wait):

    x = np.zeros((len(posns), len(freqs)))
    theta = np.zeros((len(posns), len(freqs)))
    y = np.zeros((len(posns), len(freqs)))
    fint = np.zeros((len(posns), len(freqs)))

    for i in range(0, len(posns)):

        x[:,i], theta[:,i], y[:,i], fint[:,i] = Measure(posns = posns, voltage=voltage, freq=freqs[i], tc=tcs[i], wait=wait)

    return x, theta, y, fint
'''
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
    lockin.write("PHAS 0") 
    lockin.write("SCAL 8")
    lockin.write("IRNG 3")
    lockin.write("SLVL " + str(voltage))
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
    np.savez(filename_npz, x=x, theta=theta, y=y, fint=fint, posns=positions, tcs=tc, freqs=freq)
    two_point_data = np.load(filename_npz)
    print(filename_npz)
    #print(two_point_data['x'])
    return filename_npz

#positions = TwoPointIndices(32)
#print("main positions ", positions)

V =1
Rin = 100e3
#x, theta, y, fint = Measure(np.asarray([[0,1,3,2],[7,8,10,9]]), voltage=1, freq=8000, tc=6, wait=(1/8000)*30)
RunSweep()
print("Switch status:", switch.query("S"))
ClearSwitches()

rm.close()
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