# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 12:08:52 2021
Author: Adam Coxson, MPhys Undergrad, The University of Manchester
Project: Machine Learning enabled Electrical Impedance Tomography of Graphene
Module: reconstruction_plotting
Dependancies: 
"""

import matplotlib as mpl
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from pyeit.eit.interp2d import sim2pts
import pyeit.eit.jac as jac
from datetime import datetime
import pyeit.eit.jac as jac
import greit_rec_training_set as train
from pyeit.eit.fem_for_given_meas import Forward as Forward_given
from pyeit.eit.interp2d import sim2pts
from meshing import mesh
from selection_algorithms import vdp_newton_raphson_iter, van_der_pauw_calculations

# mpl.rcParams['mathtext.fontset'] = 'cm'
# mpl.rcParams['font.family'] = 'serif'
# mpl.rcParams['font.serif'] = ['cmr10']

# mpl.rcParams['font.family'] = 'sans-serif'
# mpl.rcParams['font.sans-serif'] = ['cmr10']
#mpl.rcParams['axes.unicode_minus'] = False

def Linear_fit(x, y, yerr):
    
    s = yerr
    w = 1/(s**2)
    Aw = np.mean(w)
    Ax = np.mean(x*w)       # Calculate <x> = 1/N * sum(xi)
    Ay = np.mean(y*w)
    Axy = np.mean(x*y*w)
    Ax2 = np.mean((x**2)*w)      
    
    D = (Aw*Ax2 - Ax * Ax)                #// Expression for denominator used in m and c
    m = (Aw*Axy - Ax * Ay) / D            #// Calculate gradient m
    dm = pow(Aw / D, 0.5)                 #// Error on gradient dm
    c = (Ax2*Ay - Ax * Axy) / D          #// Calculate intercept
    dc = pow(Ax2 / D, 0.5)                #// Error on intercept dc
    
    yf = (m*x + c)                        #// Array to store new fitted y values        
        
    # Chi-squared calculator
    X2chi = 0
    yR = y - yf
    X2chi = np.sum((yR/s)**2)             # Finding the y residual values   
        
    print("Line equation: y =", m, "X \u00B1 ",dm, " +  ", c, "\u00B1", dc)
    print("χ2 = ", X2chi, ", Reduced χ2 =", X2chi/(len(x)-2) )
    
    plt.figure()
    plt.errorbar(x, y, yerr = yerr, fmt = 'bx')
    plt.plot(x, yf)
    #plt.title('Fitted values')
    plt.xlabel('ln(frequency) (Hz) ' )
    plt.ylabel('Gain (dB)')
    plt.xscale('log')
    #plt.text(np.amin(x), np.amin(y), (format(X2chi/(len(x)-2), "5.3f")) )
    plt.minorticks_on()
    plt.grid(True)
    plt.show()
    return yf, m, dm, c, dc, X2chi/(len(x)-2); 

def read_data_EIT(filename):
    positions = np.loadtxt(filename, delimiter=",", skiprows=1, usecols=[0,1,2,3])
    voltages = np.loadtxt(filename, delimiter=",", skiprows=1, usecols=4)

    return voltages, positions

def read_data_xy(filename):
    positions = np.loadtxt(filename, delimiter=",", skiprows=1, usecols=[0,1,2,3])
    x = np.loadtxt(filename, delimiter=",", skiprows=1, usecols=4)
    y = np.loadtxt(filename, delimiter=",", skiprows=1, usecols=6)

    return x,y, positions

def read_plot_freq_sweep(filename):
    f = np.loadtxt(filename, delimiter=",", skiprows=2, usecols=0)
    v = np.loadtxt(filename, delimiter=",", skiprows=2, usecols=3)
    v_err = np.loadtxt(filename, delimiter=",", skiprows=2, usecols=4)
    v_frac_err = abs(v_err/v)
    #gain = 20*np.log10(abs(v)*(1e4)) 
    gain = 20*np.log10((v)) 
    #gain = 20*np.log10(abs(v.max(v)))
    gain2 = abs(gain)-abs(np.min(gain))
    gain_err = gain*v_frac_err
    
    plt.figure()
    plt.errorbar(x=f, y=gain2, yerr=gain_err,xerr=None, fmt='rx')
    #plt.plot(f, 20*np.log10(abs(v)*(1e6)),'rx')
    plt.xlabel('Frequency (Hz)' , fontsize = 12)
    #plt.ylabel("Voltage (micro-V)", fontsize = 12)
    plt.ylabel("Gain (dB)", fontsize = 12)
    plt.xscale('log')
    #plt.yscale('log')
    plt.title("Frequency sweep: [0312] to [4545]")
    #plt.title("Slope 6 dB")
    plt.minorticks_on()
    plt.grid(b=True, which='major', axis='both')
    plt.show()
    return f, v, v_err

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

def reconstruction_plot(ex_volt_mat, measured_voltages, mesh_obj=None, title=None, n_el=32, n_per_el=3, p=0.2, lamb=0.01, n_pix=64, ref_perm=1.0, drv=1.0, mesh_params=[0.054,3000,9.34,1.89]):
    volt_mat = ex_volt_mat[:, 2:]
    ex_mat, ind = ex_mat_ind_finder(ex_volt_mat)
    if mesh_obj == None:
        mesh_obj = mesh(n_el=n_el, num_per_el=n_per_el, start_pos='left',
                        edge=0.08,el_width=0.04,mesh_params=[0.054,3000,9.34,1.89]) # Making an empty mesh
    el_pos = np.arange(n_el * n_per_el).astype(np.int16)
    fwd = Forward_given(mesh_obj, el_pos, n_el)
    f, meas, new_ind = fwd.solve_eit(volt_mat=volt_mat, ex_mat=ex_mat, new_ind=ind, drv_a=drv, drv_b=-1*drv)
    greit = train.greit.GREIT(mesh_obj, el_pos, f=f, ex_mat=ex_mat, parser='std')
    greit.setup(p=p, lamb=lamb, n=n_pix)
    reconstruction = greit.solve(measured_voltages, f.v, normalize=False).reshape(n_pix, n_pix)
    del fwd
    
    # plt.figure()
    # im = plt.imshow((reconstruction), cmap=plt.cm.viridis, origin='lower', extent=[-1, 1, -1, 1])
    # #plt.title("GREIT Reconstruction\n(Current, Voltage) pairs: ("+str(ind[-1]+1)+", "+str(len(volt_mat))+")")
    # #plt.title("Method opp-adj, p="+str(p)+", lamb="+str(lamb))
    # plt.title(title)
    # cb = plt.colorbar(im)
    # cb.ax.tick_params(labelsize=15)
    # plt.yticks(ticks=[-1.0,-0.75,-0.50,-0.25,0,0.25,0.50,0.75, 1.00], labels=[-1.0,-0.75,-0.50,-0.25,0,0.25,0.50,0.75, 1.00], fontsize=15)
    # plt.xticks(ticks=[-1.0,-0.75,-0.50,-0.25,0,0.25,0.50,0.75, 1.00], labels=[-1.0,-0.75,-0.50,-0.25,0,0.25,0.50,0.75, 1.00], fontsize=15)
    # #plt.xlabel(r'x' , fontsize = 12, fontname = 'cmr10')
    # #plt.ylabel(r'y', fontsize = 12)
    # #plt.savefig(filepath+"reconstruction")
    # plt.show()
    return reconstruction
root = os.getcwd()
folder = "\\example_data_runs\\"
# Example data from mesaurements of the uniform graphene sample and a resistor grid used to test the setup
#file = '2020-12-15-14-56-14_grid_opp-adj_8000Hz-tc6-wait160-std_p0-2_lamb0-1.csv' # Resistor Grid A
file = '2020-12-17-12-47-21_newgrid_opp-adj_8000Hz-tc6-wait160_p0-2_lamb0-1.csv' # Resistor grid B
#file = '2021-05-07-17-31-53_aesa_uniform_4_p0-5_l0-01_npel2_finemesh.csv' # Uniform Graphene sample measurements from the AESA method
#file = '2021-03-23-22-05-47_vanderpauw_frq4000_tc7_wait1000.csv' # Measurements from Van der Pauw selection algorithm to calc sheet conductivity


filename = root+folder+file
#filename2 = root+folder+file2
#filename = root+"\\"+file
filename = file


# x_row = np.arange(-0.9,1.1,0.2)
# y_row = np.ones(10)

# fig = plt.figure()
# ax = fig.gca()
# # ax.set_xticks(np.arange(-0.9, 1, 0.2))
# # ax.set_yticks(np.arange(-0.9, 1., 0.2))
# ax.set_xticks(np.arange(-1.0, 1.5, 0.5))
# ax.set_yticks(np.arange(-1.0, 1.5, 0.5))
# j = -0.9
# for i in range(0,10):
#     plt.plot(x_row, y_row*j, 'bo')
#     j = j+0.2
# plt.xlim(-1.1,1.1)
# plt.ylim(-1.1,1.1)
# #plt.grid()
# plt.show()


# mode = 'vdp'
# if mode == 'vdp':
#     voltages, resistances, positions = read_data_xy(filename)
#     resistances = abs(resistances)
#     R, s = van_der_pauw_calculations(resistances)
#     exit(1)

# simulating the cut sample
# line = {'name': 'line', 'len': 0.6, 'x': -0.3, 'y': 0.1, 'angle_line': 0.70*np.pi, 'perm': 0.00001}
# spot = {'name': 'ellipse', 'x': 0.3, 'y': 0.4, 'a': 3.5/10, 'b': 3.5/10 ,'angle': 0, 'perm': 1.2}
# anoms = np.array([line, spot], dtype='O')

n_el = 64
n_per_el = 2
#perm = 10770
perm = 11937
drv = -1.0e-4
n_pix=128
voltages, positions = read_data_EIT(filename)
#voltages2, positions2 = read_data_EIT(filename2)
#voltages = voltages*1e6 # Scaling according to measurement precision
params = [0.045,3000,3.67,3.13] # Forward mesh generation density control parameters
#params = [0.045,3000,5,5]
#params = [0.054,2500,9.34,1,89]
#params = [0.054,2500,10,10]
#params = [0.06,2500,0.45,0.3]
mesh_obj = mesh(n_el=n_el, num_per_el=n_per_el, edge=0.08,el_width=0.04,mesh_params=params,ref_perm=perm,start_pos='left') # Making an empty mesh

pts = mesh_obj['node']
tri = mesh_obj['element']
perm = mesh_obj['perm']

# volt_mat = positions[:, 2:]
# ex_mat, ind = ex_mat_ind_finder(positions)
# #mesh_obj = mesh(n_el=n_el, num_per_el=n_per_el, start_pos='left', edge=0.08,el_width=0.04,mesh_params=params) # Making an empty mesh
# el_pos = np.arange(n_el * n_per_el).astype(np.int16)
# fwd = Forward_given(mesh_obj, el_pos, n_el)
# p=0.7
# lamb = 0.01
# f, meas, new_ind = fwd.solve_eit(volt_mat=volt_mat, ex_mat=ex_mat, new_ind=ind, perm=perm)
# greit = train.greit.GREIT(mesh_obj, el_pos, f=f, ex_mat=ex_mat)
# greit.setup(p=p, lamb=lamb, n=n_pix)
# reconstruction1 = greit.solve(voltages, f.v, normalize=False).reshape(n_pix, n_pix)

# f, meas, new_ind = fwd.solve_eit(volt_mat=volt_mat, ex_mat=ex_mat, new_ind=ind, perm=10770)
# greit = train.greit.GREIT(mesh_obj, el_pos, f=f, ex_mat=ex_mat)
# greit.setup(p=p, lamb=lamb, n=n_pix)
# reconstruction2 = greit.solve(voltages2, f.v, normalize=False).reshape(n_pix, n_pix)

# reconstruction = reconstruction2-reconstruction1
# plt.figure()
# im = plt.imshow((reconstruction), cmap=plt.cm.viridis, origin='lower', extent=[-1, 1, -1, 1])
# cb = plt.colorbar(im)
# cb.ax.tick_params(labelsize=12)
# plt.yticks(ticks=[-1.0,-0.75,-0.50,-0.25,0,0.25,0.50,0.75, 1.00], labels=[-1.0,-0.75,-0.50,-0.25,0,0.25,0.50,0.75, 1.00], fontsize=12)
# plt.xticks(ticks=[-1.0,-0.75,-0.50,-0.25,0,0.25,0.50,0.75, 1.00], labels=[-1.0,-0.75,-0.50,-0.25,0,0.25,0.50,0.75, 1.00], fontsize=12)
# plt.show()
#voltages = voltages*1e20
#voltages = voltages*1e-6
p=0.2
lamb=0.01
title1 = "p="+str(p)+", lamb="+str(lamb)
rec1=reconstruction_plot(positions, voltages, mesh_obj=mesh_obj, n_el=n_el,n_per_el=n_per_el, p=p, lamb=lamb, n_pix=128, title=title1, drv=drv)
p=0.5
lamb=0.01
title2 = "p="+str(p)+", lamb="+str(lamb)
rec2=reconstruction_plot(positions, voltages, mesh_obj=mesh_obj, n_el=n_el,n_per_el=n_per_el, p=p, lamb=lamb, n_pix=128, title=title2, drv=drv)
p=0.6
lamb=0.1
title3 = "p="+str(p)+", lamb="+str(lamb)
rec3=reconstruction_plot(positions, voltages, mesh_obj=mesh_obj, n_el=n_el,n_per_el=n_per_el, p=p, lamb=lamb, n_pix=128, title=title3, drv=drv)
p=0.6
lamb=0.01
title4 = "p="+str(p)+", lamb="+str(lamb)
rec4=reconstruction_plot(positions, voltages, mesh_obj=mesh_obj, n_el=n_el,n_per_el=n_per_el, p=p, lamb=lamb, n_pix=128, title=title4, drv=drv)

fig, axs = plt.subplots(2, 2, figsize=(12, 8))  
im1 = axs[0, 0].imshow(rec1, cmap=plt.cm.viridis, origin='lower', extent=[-1, 1, -1, 1])
fig.colorbar(im1, ax=axs[0, 0])
axs[0, 0].set_title(title1)

im2 = axs[1, 0].imshow(rec2, cmap=plt.cm.viridis, origin='lower', extent=[-1, 1, -1, 1])
fig.colorbar(im2, ax=axs[1, 0]) 
axs[1, 0].set_title(title2)

im3 = axs[0, 1].imshow(rec3, cmap=plt.cm.viridis, origin='lower', extent=[-1, 1, -1, 1])
fig.colorbar(im3, ax=axs[0, 1]) 
axs[0, 1].set_title(title3)

im4 = axs[1, 1].imshow(rec4, cmap=plt.cm.viridis, origin='lower', extent=[-1, 1, -1, 1])
fig.colorbar(im4, ax=axs[1, 1]) 
axs[1, 1].set_title(title4)
fig.subplots_adjust(top=0.9)
plt.show()

p=0.5
lamb=0.01
#plt.figure("GREIT p"+str(p)+"_L"+str(lamb))
plt.figure()
im = plt.imshow((rec3), cmap=plt.cm.viridis, origin='lower', extent=[-1, 1, -1, 1])
cb = plt.colorbar(im)
cb.ax.tick_params(labelsize=12)
plt.yticks(ticks=[-1.0,-0.75,-0.50,-0.25,0,0.25,0.50,0.75, 1.00], labels=[-1.0,-0.75,-0.50,-0.25,0,0.25,0.50,0.75, 1.00], fontsize=12)
plt.xticks(ticks=[-1.0,-0.75,-0.50,-0.25,0,0.25,0.50,0.75, 1.00], labels=[-1.0,-0.75,-0.50,-0.25,0,0.25,0.50,0.75, 1.00], fontsize=12)
plt.show()

lamb=0.01
p=0.63
volt_mat = positions[:, 2:]
ex_mat, ind = ex_mat_ind_finder(positions)
el_pos = np.arange(n_el * n_per_el).astype(np.int16)
fwd = Forward_given(mesh_obj, el_pos, n_el)
f, meas, new_ind = fwd.solve_eit(volt_mat=volt_mat, new_ind=ind, ex_mat=ex_mat, drv_a=drv, drv_b=drv) # forward solve on the empty mesh
eit_gn = jac.JAC(mesh_obj, el_pos, ex_mat=ex_mat, f=f, perm=perm, parser='std')
# parameter tuning is needed for better EIT images
eit_gn.setup(p=p, lamb=lamb, method='kotre')
sigma_gn_raw = eit_gn.gn(voltages, maxiter=20, gtol=1e-4, p=p, lamb=lamb, method='kotre',verbose=True,
                          mesh=mesh_obj, fwd_in=fwd,volt_mat=volt_mat, ind=ind, ex_mat=ex_mat)
sigma_gn = sim2pts(pts, tri, sigma_gn_raw)
      
#plt.figure("Gauss-Newton p="+str(p)+", lamb="+str(lamb))
plt.figure()
im_gn = plt.tripcolor(pts[:, 0], pts[:, 1], tri, np.real(sigma_gn), cmap=plt.cm.viridis)
cb=plt.colorbar(im_gn)
cb.ax.tick_params(labelsize=12)
plt.xlim(-1.,1.0)
plt.ylim(-1.,1.0)
plt.yticks(ticks=[-1.0,-0.75,-0.50,-0.25,0,0.25,0.50,0.75, 1.00], labels=[-1.0,-0.75,-0.50,-0.25,0,0.25,0.50,0.75, 1.00], fontsize=12)
plt.xticks(ticks=[-1.0,-0.75,-0.50,-0.25,0,0.25,0.50,0.75, 1.00], labels=[-1.0,-0.75,-0.50,-0.25,0,0.25,0.50,0.75, 1.00], fontsize=12)
#plt.title("Gauss-Newton p="+str(p)+", lamb="+str(lamb))
plt.show()

# lamb=0.01
# p=0.5
# # parameter tuning is needed for better EIT images
# eit_gn.setup(p=p, lamb=lamb, method='kotre')
# sigma_gn_raw = eit_gn.gn(voltages, maxiter=10, gtol=1e-4, p=p, lamb=lamb, method='kotre')
# sigma_gn = sim2pts(pts, tri, sigma_gn_raw)
# fin_time = datetime.now()
# print("Gauss-Newton time:", fin_time-start_time)                
# #plt.figure("Gauss-Newton p="+str(p)+", lamb="+str(lamb))
# plt.figure()
# im_gn = plt.tripcolor(pts[:, 0], pts[:, 1], tri, np.real(sigma_gn), cmap=plt.cm.viridis)
# plt.colorbar(im_gn)
# #plt.title("Gauss-Newton p="+str(p)+", lamb="+str(lamb))
# plt.show()

# lamb=0.01
# p=0.62
# # parameter tuning is needed for better EIT images
# eit_gn.setup(p=p, lamb=lamb, method='kotre')
# sigma_gn_raw = eit_gn.gn(voltages, maxiter=10, gtol=1e-4, p=p, lamb=lamb, method='kotre')
# sigma_gn = sim2pts(pts, tri, sigma_gn_raw)
# fin_time = datetime.now()
# print("Gauss-Newton time:", fin_time-start_time)                
# #plt.figure("Gauss-Newton p="+str(p)+", lamb="+str(lamb))
# plt.figure()
# im_gn = plt.tripcolor(pts[:, 0], pts[:, 1], tri, np.real(sigma_gn), cmap=plt.cm.viridis)
# plt.colorbar(im_gn)
# plt.title("Gauss-Newton p="+str(p)+", lamb="+str(lamb))
# plt.show()

#del mesh_obj


#############################################################################
# f, v, v_err = read_plot_freq_sweep(filename)
# C = -115.683
# plt.axhline(y = C, color = 'b', linestyle = '-')
# slope_f = np.array([1e3,1e4,1e5])*3.5*10
# slope_f2 = np.array([1e3,1e4,1e5])*10
# slope_y = np.array([0,-20,-40])+C
# plt.plot(slope_f,slope_y, 'b-')
# plt.plot(slope_f2,slope_y, 'b-')

# gain = 20*np.log10(abs(v)) 
# gain = abs(gain)-abs(np.min(gain)) # normalise to zero
# gain_err = gain*(v_err/v)

# Fitted_y, Gradient, Grad_err, Intercept, Itcp_err, chisq_red = Linear_fit(np.log(f[-10:]),gain[-10:],gain_err[-10:]) # Calling the linear fit function to apply least squares fit
# cutoff_y=-3.01
# R=2.09
# X = ((cutoff_y-Intercept)/Gradient)
# fc=np.exp(X)
# fc_frac_err = ((Itcp_err/Intercept)**2+(Grad_err/Gradient)**2)**0.5 
# fc_err=fc_frac_err*fc
# C=1/(2*np.pi*fc*R)
# C_err = C*fc_frac_err

# plt.figure()
# plt.scatter(f[0:11],gain[0:11])
# p = np.polyfit(np.log(f[0:11]), gain[0:11], 1)
# plt.semilogx(f[0:11], p[0] * np.log(f[0:11]) + p[1], 'g--')
# plt.xscale('log')
# plt.show()


#############################################################################
# x,y, positions = read_data_xy_contact_check(filename)
# voltages_x = np.zeros((64,64))
# voltages_y = np.zeros((64,64))
# theta      = np.zeros((64,64))
# for i in range(0,len(x)):
#     voltages_x[int(positions[i][0])][int(positions[i][1])]=x[i]
#     voltages_y[int(positions[i][0])][int(positions[i][1])]=y[i]
    
# theta=np.arctan(voltages_y/voltages_x)*(360/(2*np.pi))
# y_over_x = abs(voltages_y/voltages_x)


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

# plt.figure()
# im_yoverx = plt.imshow(y_over_x,cmap='viridis',interpolation='nearest', origin='lower')
# plt.xlabel('Contact number' , fontsize = 12, fontname = 'cmr10')
# plt.ylabel('Contact number', fontsize = 12, fontname = 'cmr10')
# plt.title("| y/x |")
# plt.tick_params(axis='both', which='major', labelsize=12)
# plt.minorticks_on()
# plt.colorbar(im_yoverx)
# plt.show()


