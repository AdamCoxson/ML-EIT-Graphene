# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 10:14:40 2020
Authors: Frederik Brookebarnes and Adam Coxson, MPhys Undergrads, The University of Manchester
Project: Automated Electrical Impedance Tomography of Graphene
Module:  GREIT_reconstruction.py
Dependancies: 
"""
import numpy as np
import matplotlib.pyplot as plt
import csv

import pyeit.mesh.vanilla as mesh
#from pyeit.mesh.wrapper_original import create, set_perm, layer_circle
from pyeit.mesh.vanilla.shape import rectangle
from pyeit.eit.fem_original import Forward
from pyeit.eit.utils import eit_scan_lines
from  pyeit.eit.interp2d import sim2pts # for Jacobian plot
import pyeit.eit.greit as greit
import pyeit.eit.bp as bp
import pyeit.eit.jac as jac


from datetime import datetime

def electrode_posns(n_el, n_sides=4):
    """
    Calculate fixed electrode positions for a regular shape.
    
    Parameters
    ----------
    n_el: (int) Total number of electrodes
    n:    (int) Number of sides, default 4 for square sample
    
    Returns
    -------
    electrode_positions (array (n_el, 2)): An array containing x,y coordinates
        for fixed electrode positions. CW along perimeter from 9 o'clock pos.  

    Notes
    -----
    This produces and array containing the (x,y) values for regularly spaced 
    fixed electrode positions for a regular 2D shape. It ensures the first 
    electrode is at the 9 o'clock point on the perimeter of the sample and 
    continues clockwise for subsequent electrode numbers.
   
    WARNING
    -------
    Check the electrodes do follow the samples perimeter in a 
    clockwise manner. 
    TO DO: Account for an even/odd no. of electrodes per side. Use an if statement.
    """
    # Getting equally spaced co-ords
    posns = (np.arange(1,(n_el/n_sides+1)) * 2/(n_el/n_sides+1)) - 1 
    ones = np.ones(np.size(posns))
    top = np.array((posns, ones)).T # top row of electrode positions
    bottom = np.array((-posns, -ones)).T # +/- signs ensure CW allocation
    left = np.array((-ones, posns)).T
    right = np.array((ones, -posns)).T
    args = (left,top,right,bottom) # ordered CW from centre of left side.
    electrode_positions = np.concatenate(args)
    
    shift = -4 # Hardcoded for 8 electrodes per side, need to account for even and odd n_el shifting
    electrode_positions = np.roll(electrode_positions, shift, axis=0) # Shift to 9 o'clock position
    return electrode_positions

def read_data(filepath):
    electrodes = np.array([])
    voltages = []
    skipfirstline = True
    with open(filepath) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            line = np.array([row[0],row[1],row[2],row[3]])
            if skipfirstline == True:
                skipfirstline=False
                continue
            if len(electrodes)==0:
                electrodes = line
            else:
                electrodes = np.vstack((electrodes, line))
            voltages.append(float(row[4]))

    return electrodes, voltages

def read_LTspice_data(n_files=32):
    #path = 'LTspice Voltage data\\low right asymmetric grid\\' # path is myCWD\\LTspice ... 
    #path = 'LTspice Voltage data\\uniform grid\\csv\\'
    #path = 'LTspice Voltage data\\Incorrectly ordered data\\uniform_r_grid\\'
    #path = 'LTspice Voltage data\\Incorrectly ordered data\\low right asymmetric grid\\'
    
    #path = 'LTspice Voltage data\\10k_bottomright\\csv\\'
    path = 'LTspice Voltage data\\uniformmk2\\csv\\'

    voltages = []
    for i in range(0, n_files):
        filename = path+str(i)+'.csv'
        voltages = np.concatenate((voltages,np.genfromtxt(filename,delimiter=',')),axis=None)
    return voltages

file = '2020-12-01-20-20-56_resitor_grid_8000Hz-tc10-wait1000-step16-dist1.csv'
el, v = read_data(file)
v = np.array(v)


n_electrodes = 32  # Square mesh with 8 electrodes per side
el_posns = electrode_posns(n_electrodes)
#new_voltages = read_data(n_electrodes)
init_time_A = datetime.now()

""" 0. construct mesh """
def _fd(pts):
    rect = rectangle(pts, p1=[-1, -1], p2=[1, 1])
    return rect
mesh_obj, el_pos, p_fix = mesh.create(n_el=n_electrodes, fd=_fd, fh=None, p_fix=el_posns, bbox=None, h0=0.1)

# extract node, element, alpha
pts = mesh_obj['node']
tri = mesh_obj['element']

plot_mesh = 0         # plot and show the meshes
if (plot_mesh == 1):
    fig, ax = plt.subplots()
    ax.triplot(pts[:, 0], pts[:, 1], tri)
    ax.plot(p_fix[:, 0], p_fix[:, 1], 'ro')
    ax.set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    title_src = 'number of triangles = ' + str(np.size(tri, 0)) + ', ' + \
                'number of nodes = ' + str(np.size(pts, 0))
    plt.title(title_src)
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    plt.show()
    
""" 1. problem setup """
# test function for altering the 'permittivity' in mesh
anomaly = [{'x': 0.4,  'y': 0,    'd': 0.4, 'perm': 10},
           {'x': -0.4, 'y': 0,    'd': 0.2, 'perm': 0.1},
           {'x': 0,    'y': 0.5,  'd': 0.1, 'perm': 10},
           {'x': 0,    'y': -0.5, 'd': 0.1, 'perm': 0.1}]
mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=1.0)
delta_perm = np.real(mesh_new['perm'] - mesh_obj['perm'])

plot_cond = 0 # Plot absolute conductivity heat map
if (plot_cond == 1):
    fig, ax = plt.subplots()
    im = ax.tripcolor(pts[:, 0], pts[:, 1], tri, delta_perm,
                       shading='flat', cmap=plt.cm.viridis)
    fig.colorbar(im)
    ax.axis('equal')
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_title(r'$\Delta$ Conductivity')

""" 2. FEM forward simulations """
# setup EIT scan conditions
el_dist, step = 1, 16
ex_mat = eit_scan_lines(n_electrodes, el_dist)

# calculate simulated data
fwd = Forward(mesh_obj, el_pos)
f0,volt_mat_0 = fwd.solve_eit(ex_mat, step=step, perm=mesh_obj['perm'])
f1,volt_mat_1 = fwd.solve_eit(ex_mat, step=step, perm=mesh_new['perm'])
init_time_B = datetime.now()

""" 3.1 Construct using GREIT """
eit = greit.GREIT(mesh_obj, el_pos,f=f0, ex_mat=ex_mat, step=step, parser='std')
p       = 0.2
lamb    = 0.5
scaling = int(1e4)
eit.setup(p=p, lamb=lamb, n = 128)
volt_zero = np.zeros(len(v))
#ds = eit.solve(v*scaling, volt_zero)
#ds = eit.solve(f1.v, f0.v)
ds = eit.solve(v*scaling, f0.v)
x, y, ds = eit.mask_value(ds, mask_value=np.NAN)

cmap = plt.cm.viridis
fig, ax = plt.subplots()
im = ax.imshow(np.real(ds), interpolation='none', cmap=cmap, extent=[-1.0,1.0,-1.0,1.0], origin='lower')
ax.plot(p_fix[:, 0], p_fix[:, 1], 's', markersize = 3.0, color = 'black')
fig.colorbar(im)
title = 'GREIT p='+str(p)+', l='+str(lamb)+', scaling='+str(scaling)
ax.set_title(title)
ax.axis('equal')
ax.set_aspect('equal')
plt.show()