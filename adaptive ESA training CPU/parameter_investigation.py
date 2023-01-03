# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 16:25:38 2021
Author: Adam Coxson, MPhys Undergrad, The University of Manchester
Project: Machine Learning enabled Electrical Impedance Tomography of Graphene
Module: electrode_corner_investigation
Dependancies: pyeit.eit: .fem_for_given_meas .fem_forallmeas .utils, greit_rec_training_set, measurement_optimizer
              get_next_prediction.py, meshing.py

A script created for testing several different things in the main:
    1) anomaly file generation and reconstruction
    2) Greit reconstructions for a varied set of edge and contact widths
    3) Greit reconstruction for a single set of setup parameters
    4) Reconstructions using different methods
    5) Bayesian Optimisation of different mesh parameters that define mesh density and internodal distance
    6) Further evaluation and fine-tuning of promising mesh parameters
"""

# Generic package imports
import os
import numpy as np
import cupy as cp # CUDA GPU accelerated python
import matplotlib.pyplot as plt
import pickle     # For saving/loading anomaly classes
import csv   

# Imports for Bayesian optimisation
import warnings
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from datetime import datetime
from joblib import Parallel, delayed

# Modules
import pyeit.eit.jac as jac
import greit_rec_training_set as train
from pyeit.eit.fem_for_given_meas import Forward as Forward_given
from pyeit.eit.interp2d import sim2pts
from meshing import mesh
from get_next_prediction import initialise_ex_volt_mat, getNextPrediction_partialforwardsolver


# def mesh_plot(mesh_obj):
#     p = mesh_dict['p']
#     pfix = mesh_dict['pfix']
#     fig, ax = plt.subplots()
#     ax.triplot(p[:, 0], p[:, 1], t)
#     ax.set_aspect('equal')
#     ax.set_xlim([-1.2, 1.2])
#     ax.set_ylim([-1.2, 1.2])
#     plt.yticks(ticks=[-1.0,-0.50,0,0.50, 1.00], labels=[-1.0,-0.50,0,0.50,1.00], fontsize=12)
#     plt.xticks(ticks=[-1.0,-0.50,0,0.50, 1.00], labels=[-1.0,-0.50,0,0.50,1.00], fontsize=12)
#     ax.plot(el_pos[:, 0], el_pos[:, 1], 's', markersize = 2.0, color = 'red')
#     plt.show()

def lossL2(im, true):
	return np.sum((im - true)**2)

def single_evaluation(fwd, mesh_obj, greit, ref_voltages, volt_mat, ind, ex_mat, i):
    load_name = 'anomaly_'+str(i)
    try:
        with open(str(load_name)+'.pkl','rb') as load_file:
            anomaly = pickle.load(load_file)
    except:
        print("Save file not found")
        exit(1)
    n_pix=128
    true = train.generate_examplary_output(2.0, int(n_pix), anomaly) # true conductivty map
    mesh_new = train.set_perm(mesh_obj, anomaly=anomaly, background=1) # New mesh with anomalies
    f_sim, meas_sim, ind_sim = fwd.solve_eit(volt_mat=volt_mat, new_ind=ind, ex_mat=ex_mat, perm=mesh_new['perm'].astype('f8'))
    reconstruction = greit.solve(f_sim.v, ref_voltages).reshape(n_pix, n_pix)
    L2loss = lossL2(reconstruction, true)
    return L2loss
def EIT_evaluation(cfg, n_anomalies=10):

    # Format of cfg variable is [h0, maxiter, a_coeff, b_coeff]
    a = 2.0
    n_el     = 64
    n_per_el = 2
    n_pix    = 128
    edge = 0.04*a
    el_width = 0.02*a
    #anomaly_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    results = np.zeros(n_anomalies)
    ex_volt_mat, ex_mat, volt_mat, ind = initialise_ex_volt_mat(current_mode='opp',volt_mode='adj',n_el=n_el,ex_mat_length=None)
    el_pos = np.arange(n_el * n_per_el).astype(np.int16)
    
    mesh_obj = mesh(n_el=n_el,num_per_el=n_per_el, edge=edge, el_width=el_width, mesh_params=cfg) # Making an empty mesh
    fwd = Forward_given(mesh_obj, el_pos, n_el)
    f, meas, new_ind = fwd.solve_eit(volt_mat=volt_mat, new_ind=ind, ex_mat=ex_mat) # forward solve on the empty mesh
    ref_voltages = f.v
    greit = train.greit.GREIT(mesh_obj, el_pos, f=f, ex_mat=ex_mat)
    greit.setup(p=p, lamb=lamb, n=n_pix)
    
    for i in range(0,n_anomalies):
        results[i] = single_evaluation(fwd, mesh_obj, greit, ref_voltages, volt_mat, ind, ex_mat, i+1)
    return np.array(results)

def mesh_parameter_evaluation(h0, maxiter, a_coeff, b_coeff, n_repeats=20, n_anomalies=20):
        """
        This function is defined here to be compatible with gp_minimize, below the 'bounds' variable and @use_named_args.
        A single output is used to pass straight into the Bayesian optimisation. 
        """
        anomaly_results = np.zeros(shape=(n_repeats,n_anomalies))
        cfg = [h0, maxiter, a_coeff, b_coeff]
        global counter
        counter = counter + 1
        try:
            for i in range(0, n_repeats):
                anomaly_results[i][:] = EIT_evaluation(cfg, n_anomalies=n_anomalies)
            means = np.mean(anomaly_results, axis=0)
            stds = np.std(anomaly_results, axis=0)
            error_output = np.sum(means) + np.sum(0.5*stds)
            #print(anomaly_results)
        except:
            error_output = 1.0e4
        print_vals = True
        if print_vals == True:
            print('%d) output: %d, h0: %.4f, maxiter: %d, a_coeff: %.4f, b_coeff: %.4f' % (counter,error_output,cfg[0], cfg[1],cfg[2],cfg[3]))
        return error_output # Return the average mse plus some of its error. Biases gp_minimize to use smaller stds

def write_to_csv(filename, outputs, x_iters, hyperparameter_names, n_samples, time, cores):
    """
    ARGS: filename: Name of the csv file to write the data to,
          results: a list of differnet MSE results for all hyperparameter sets,
          x_iters: the values for each hyperparameter passed in using the 'bounds' variable,
          hyperparameter_names: A list of all the hyperparameters used,
          n_samples: number of iterations of training and testing which were then averaged,
          time: total script execution time,
          cores: number of cores used to do multiple iterations of n_samples simultaneously,
    OUTPUTS: none, writes a csv file to cwd.
    
    This function takes in all the necessary details you would want recorded to use in later analysis
    """
    if type(x_iters)==np.ndarray:
        x_iters = np.insert(x_iters,0,outputs,axis=1)
    else:
        for i in range(0,len(outputs)):
            x_iters[i].insert(0,outputs[i]) 
    
    writer = csv.writer(open(filename,'w'),lineterminator ='\n')
    writer.writerow(["Time:", time, "Samples",n_samples, "Cores:",cores])
    writer.writerow(hyperparameter_names)
    writer.writerows(x_iters)
    print("Written to",filename,"successfully.")
    return None

if __name__ == "__main__":
    n_el = 128
    n_per_el = 2
    n_pix = 64
    current_mode = 'opp'
    volt_mode = 'adj'
    ESA_volt_mode = 'all' # voltage pairs which the jacobian can try to calculate
    #ex_mat_length = 10
    p = 0.5
    lamb = 0.1
    a = 2.0 # side length
    save_plots = False
    num_anomalies = 10
    mode = 2
    
    
    if (mode == 1): # Anomaly file generation
        print("Simulating anomalies and voltage data")
        anomaly_list=np.zeros(num_anomalies).astype(object)
        for i in range(0,num_anomalies):
            anomaly = train.generate_anoms(a, a)
            anomaly_list[i]=anomaly
            true = train.generate_examplary_output(a, int(n_pix), anomaly) # true conductivty map
        
            plt.figure()
            im1 = plt.imshow(true, cmap=plt.cm.viridis, origin='lower', extent=[-1, 1, -1, 1])
            plt.colorbar(im1)
            plt.title("True Image "+str(i))
            plt.show()
            
        anom_choice = int(input("Select index of desired anomaly [0 to 4]:"))
        anomaly_to_save = anomaly_list[anom_choice]
        save_name = 'anomaly_20'
        
        with open(str(save_name)+'.pkl','wb') as save_file:
            pickle.dump(anomaly_to_save, save_file)
        
    elif (mode == 2): # GREIT evaluation for varied edge and contact widths
        load_name = 'anomaly_5'
        
        try:
            with open(str(load_name)+'.pkl','rb') as load_file:
                anomaly_test = pickle.load(load_file)
        except:
            print("Saved file not found")
            
        # Initialise current voltage excitation matrix to simulate some voltages measurements to kick start the ESA
        ex_volt_mat, ex_mat, volt_mat, ind = initialise_ex_volt_mat(current_mode=current_mode, 
                                                                    volt_mode=volt_mode,n_el=n_el, ex_mat_length=None)
        #anomaly_test = train.generate_anoms(a, a)
        true = train.generate_examplary_output(a, int(n_pix), anomaly_test) # true conductivty map
        el_pos = np.arange(n_el * n_per_el).astype(np.int16)
        #for i in [0.02, 0.05, 0.08,  0.1]: # corner-to-contact fractional edge width
        #for i in [0.02, 0.03, 0.04, 0.1]:
        #for i in [0.04, 0.05]:
            # edge = i*a # normalize w.r.t side length
            # el_width = 0.01*a
        for i in [0.01, 0.02, 0.03, 0.04]:
            edge = 0.04*a # normalize w.r.t side length
            el_width = i*a
            mesh_obj = mesh(n_el=n_el,num_per_el=n_per_el, edge=edge, el_width=el_width) # Making an empty mesh
            mesh_new = train.set_perm(mesh_obj, anomaly=anomaly_test, background=1) # New mesh with anomalies
            fwd = Forward_given(mesh_obj, el_pos, n_el)
            
            f, meas, new_ind = fwd.solve_eit(volt_mat=volt_mat, new_ind=ind, ex_mat=ex_mat) # forward solve on the empty mesh
            f_sim, meas_sim, ind_sim = fwd.solve_eit(volt_mat=volt_mat, new_ind=ind, ex_mat=ex_mat, perm=mesh_new['perm'].astype('f8'))
            greit = train.greit.GREIT(mesh_obj, el_pos, f=f, ex_mat=ex_mat)
            greit.setup(p=p, lamb=lamb, n=n_pix)
            reconstruction = greit.solve(f_sim.v, f.v).reshape(n_pix, n_pix)
            plt.figure()
            im = plt.imshow(reconstruction, cmap=plt.cm.viridis, origin='lower', extent=[-1, 1, -1, 1])
            plt.title("Contact width: "+str(el_width)+", ("+str(round(100*el_width/a,2))+"%)\nEdge width: "+str(edge)+", ("+str(round(100*edge/a,2))+"%)")
            plt.colorbar(im)
            if save_plots==True: 
                filepath = os.getcwd() + "\\data\\plots from edge investigation\\anomaly_4\\"
                plt.savefig(filepath+"h0p06_Contact_edge")
            plt.show()
            print("Loss-squared error: ", lossL2(reconstruction, true))
        
        
        plt.figure()
        im1 = plt.imshow(true, cmap=plt.cm.viridis, origin='lower', extent=[-1, 1, -1, 1])
        plt.colorbar(im1)
        plt.title("True Image")
        if save_plots==True: 
            plt.savefig(filepath+"True image")
        plt.show()
        
    elif (mode==3): # Greit reconstruction for a single set of setup parameters
        load_name = 'anomaly_10'
        
        try:
            with open(str(load_name)+'.pkl','rb') as load_file:
                anomaly_test = pickle.load(load_file)
        except:
            print("Saved file not found")
        # Initialise current voltage excitation matrix to simulate some voltages measurements to kick start the ESA
        ex_volt_mat, ex_mat, volt_mat, ind = initialise_ex_volt_mat(current_mode=current_mode, 
                                                                    volt_mode=volt_mode,n_el=n_el, ex_mat_length=None)
        #anomaly_test = train.generate_anoms(a, a)
        true = train.generate_examplary_output(a, int(n_pix), anomaly_test) # true conductivty map
        el_pos = np.arange(n_el * n_per_el).astype(np.int16)
        edge = 0.04*a # normalize w.r.t side length
        el_width = 0.02*a
        #Mesh params = h0, maxiter, a_coeff, b_coeff
        mesh_params = [0.056,1438, 2.47, 1.38]
        mesh_obj = mesh(n_el=n_el,num_per_el=n_per_el, edge=edge, el_width=el_width, mesh_params=mesh_params) # Making an empty mesh
        mesh_new = train.set_perm(mesh_obj, anomaly=anomaly_test, background=1) # New mesh with anomalies
        fwd = Forward_given(mesh_obj, el_pos, n_el)
        
        f, meas, new_ind = fwd.solve_eit(volt_mat=volt_mat, new_ind=ind, ex_mat=ex_mat) # forward solve on the empty mesh
        f_sim, meas_sim, ind_sim = fwd.solve_eit(volt_mat=volt_mat, new_ind=ind, ex_mat=ex_mat, perm=mesh_new['perm'].astype('f8'))
        greit = train.greit.GREIT(mesh_obj, el_pos, f=f, ex_mat=ex_mat)
        greit.setup(p=p, lamb=lamb, n=n_pix)
        reconstruction = greit.solve(f_sim.v, f.v).reshape(n_pix, n_pix)
        plt.figure()
        im = plt.imshow(reconstruction, cmap=plt.cm.viridis, origin='lower', extent=[-1, 1, -1, 1])
        plt.title("h0=0.05, 5.0+2.0*")
        plt.colorbar(im)
        if save_plots==True: 
            filepath = os.getcwd() + "\\data\\plots from edge investigation\\anomaly_4\\"
            plt.savefig(filepath+"h0p06_Contact_edge")
        plt.show()
        print("Loss-squared error: ",lossL2(reconstruction, true))
        
        plt.figure()
        im1 = plt.imshow(true, cmap=plt.cm.viridis, origin='lower', extent=[-1, 1, -1, 1])
        plt.colorbar(im1)
        plt.title("True Image")
        if save_plots==True: 
            plt.savefig(filepath+"True image")
        plt.show()
        
    elif (mode==4): # Reconstructions using different methods
        load_name = 'anomaly_12'
        
        try:
            with open(str(load_name)+'.pkl','rb') as load_file:
                anomaly_test = pickle.load(load_file)
        except:
            print("Saved file not found")
        # Initialise current voltage excitation matrix to simulate some voltages measurements to kick start the ESA
        ex_volt_mat, ex_mat, volt_mat, ind = initialise_ex_volt_mat(current_mode=current_mode, 
                                                                    volt_mode=volt_mode,n_el=n_el, ex_mat_length=None)
        #anomaly_test = train.generate_anoms(a, a)
        true = train.generate_examplary_output(a, int(n_pix), anomaly_test) # true conductivty map
        el_pos = np.arange(n_el * n_per_el).astype(np.int16)
        edge = 0.04*a # normalize w.r.t side length
        el_width = 0.02*a
        
        #Mesh params = h0, maxiter, a_coeff, b_coeff
        #mesh_params = [0.056, 1438, 2.47, 1.38]
        #mesh_params = [0.054, 3000, 9.34, 1.89]
        mesh_params = [0.045, 2000, 3.67, 3.13]
        mesh_obj = mesh(n_el=n_el, num_per_el=n_per_el, edge=edge, el_width=el_width, mesh_params=mesh_params) # Making an empty mesh
        pts = mesh_obj['node']
        tri = mesh_obj['element']
        mesh_new = train.set_perm(mesh_obj, anomaly=anomaly_test, background=1.) # New mesh with anomalies
    
        fwd = Forward_given(mesh_obj, el_pos, n_el)
        f, meas, new_ind = fwd.solve_eit(volt_mat=volt_mat, new_ind=ind, ex_mat=ex_mat) # forward solve on the empty mesh
        f_sim, meas_sim, ind_sim = fwd.solve_eit(volt_mat=volt_mat, new_ind=ind, ex_mat=ex_mat, perm=mesh_new['perm'].astype('f8'))

        
        # GREIT
        do_greit = True
        if do_greit == True:
            start_time = datetime.now()
            greit = train.greit.GREIT(mesh_obj, el_pos, f=f, ex_mat=ex_mat)
            greit.setup(p=p, lamb=lamb, n=n_pix)
            ds_greit = greit.solve(f_sim.v, f.v).reshape(n_pix, n_pix)
            fin_time = datetime.now()
            print("GREIT time:", fin_time-start_time)

            
            # plt.figure()
            # gr_max = np.max(np.abs(ds_greit))
            # #im2 = ax2.imshow(np.real(ds_greit), interpolation='nearest', cmap=plt.cm.viridis, vmin=-gr_max, vmax=gr_max)
            # im2 = plt.imshow(np.real(ds_greit), interpolation='nearest', cmap=plt.cm.viridis, origin='lower', extent=[-1, 1, -1, 1])
            # plt.colorbar(im2)
            # plt.title("GREIT")
            # plt.show()
        
        # JAC
        do_jac = True
        if do_jac == True:
            start_time = datetime.now()
            eit = jac.JAC(mesh_obj, el_pos, ex_mat=ex_mat, f=f, perm=1., parser='std')
            # parameter tuning is needed for better EIT images
            eit.setup(p=p, lamb=lamb, method='kotre')
            # if the jacobian is not normalized, data may not to be normalized too.
            ds_jac_raw = eit.solve(f_sim.v, f.v, normalize=False)
            ds_jac = sim2pts(pts, tri, ds_jac_raw)
            fin_time = datetime.now()
            print("JAC time:", fin_time-start_time)
            
            # plt.figure()
            # jac_max = np.max(np.abs(ds_jac))
            # #im1 = ax1.tripcolor(pts[:, 0], pts[:, 1], tri, np.real(ds_jac), cmap=plt.cm.viridis, vmin=-jac_max, vmax=jac_max)
            # im1 = plt.tripcolor(pts[:, 0], pts[:, 1], tri, np.real(ds_jac), cmap=plt.cm.viridis)
            # plt.colorbar(im1)
            # plt.title("JAC")
            # plt.show()
            
                        # Gauss-Newton
            do_gn = True
            if do_gn == True:
                start_time = datetime.now()
                eit_gn = jac.JAC(mesh_obj, el_pos, ex_mat=ex_mat, f=f, perm=1., parser='std')
                # parameter tuning is needed for better EIT images
                eit_gn.setup(p=p, lamb=lamb, method='kotre')
                sigma_gn_raw = eit_gn.gn(f_sim.v, x0=None, maxiter=1, gtol=1e-4, p=None, lamb=None,lamb_decay=1.0, lamb_min=0, method='kotre', verbose=True)
                sigma_gn = sim2pts(pts, tri, sigma_gn_raw)
                fin_time = datetime.now()
                print("Gauss-Newton time:", fin_time-start_time)
                
                # plt.figure()
                # #gn_max = np.max(np.abs(ds_jac))
                # im_gn = plt.tripcolor(pts[:, 0], pts[:, 1], tri, np.real(sigma_gn), cmap=plt.cm.viridis)
                # plt.colorbar(im_gn)
                # plt.title("Gauss-Newton")
                # plt.show()
            
            # # True
            # plt.figure()
            # imT = plt.imshow(true, cmap=plt.cm.viridis, origin='lower', extent=[-1, 1, -1, 1])
            # plt.colorbar(imT)
            # plt.title("True Image")
            # plt.show()

            fig, axs = plt.subplots(2, 2, figsize=(12, 8))  
            im1 = axs[0, 0].imshow(true, cmap=plt.cm.viridis, origin='lower')
            fig.colorbar(im1, ax=axs[0, 0])
            axs[0, 0].set_title("True")
            
            cmap = plt.get_cmap('viridis')
            im2 = axs[1, 0].imshow(np.real(ds_greit), cmap=cmap, origin='lower')
            fig.colorbar(im2, ax=axs[1, 0]) 
            axs[1, 0].set_title("GREIT")
            #plt.rcParams['image.cmap'] = 'BlueRed3'
            
            im_gn = axs[0,1].tripcolor(pts[:, 0], pts[:, 1], tri, np.real(sigma_gn), cmap=plt.cm.viridis)
            fig.colorbar(im_gn, ax=axs[0, 1], cmap=plt.cm.viridis)
            axs[0, 1].set_title("Gauss-Newton")

            im4 = axs[1,1].tripcolor(pts[:, 0], pts[:, 1], tri, np.real(ds_jac), cmap=plt.cm.viridis)
            fig.colorbar(im4, ax=axs[1, 1])
            im4.set_cmap('viridis')
            axs[1, 1].set_title("JAC")
            
            fig.subplots_adjust(top=0.9)
            plt.show()
        
    
        
    elif (mode==5): # Bayesian Optimisation of different mesh parameters that define mesh density and internodal distance
        warnings.filterwarnings("ignore", category=FutureWarning)
        #warnings.filterwarnings("ignore", category=UserWarning)
        """
        This filters the following warnings:
        C:\ProgramData\Anaconda3\lib\site-packages\torch\storage.py:34: FutureWarning: pickle support for Storage will be removed in 1.5. Use `torch.save` instead
        warnings.warn("pickle support for Storage will be removed in 1.5. Use `torch.save` instead", FutureWarning)
        C:\ProgramData\Anaconda3\lib\site-packages\joblib\externals\loky\process_executor.py:691: UserWarning: A worker 
        stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak
        """
    
        random_vals = 20
        num_evaluations = random_vals + 30
        # random_vals = 5                 # FOR DEBUG
        # num_evaluations = random_vals + 10
        # random_vals = 30
        # num_evaluations = random_vals + 30
        cores = 8
        hyperparam_idx = 0
        n_repeats=4
        anomaly_results = np.zeros(shape=(n_repeats,10))
        counter = 0
    
        # bounds = [                           
        #       #Real(0.040, 0.12,  name='h0'),  # DEBUG BOUNDS FOR QUICK RUNNING
        #       Categorical([0.04, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.09, 0.1], name = 'h0'),
        #       Integer(1000, 2000, name = 'maxiter'),
        #       Real(0.1,3.0,  name='a_coeff'),
        #       Real(0.1, 4.0, name='b_coeff')]
        # bounds = [                           
        #       #Real(0.040, 0.12,  name='h0'),  # DEBUG BOUNDS FOR QUICK RUNNING
        #       Categorical(np.linspace(0.03,0.1,36), name = 'h0'),
        #       Categorical([1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000], name = 'maxiter'),
        #       Categorical(np.linspace(0.1,5.0,50),  name='a_coeff'),
        #       #Categorical(np.linspace(0.1,5.0,50),  name='b_coeff')]
        #       Categorical(np.linspace(1.0,5.0,201), name='b_coeff')]
        bounds = [                           
              Categorical(np.linspace(0.03,0.1,71), name = 'h0'),
              Integer(400, 3000, name = 'maxiter'),
              Categorical(np.linspace(0.1,10.0,991),  name='a_coeff'),
              Categorical(np.linspace(0.02,6.0,599), name='b_coeff')]
    
        @use_named_args(bounds)
        def mesh_bayesian_evaluation(h0, maxiter, a_coeff, b_coeff):
            """
            This function is defined here to be compatible with gp_minimize, below the 'bounds' variable and @use_named_args.
            A single output is used to pass straight into the Bayesian optimisation. 
            """
            cfg = [h0, maxiter, a_coeff, b_coeff]
            global counter
            counter = counter + 1
            try:
    
                for i in range(0, n_repeats):
                    anomaly_results[i][:] = EIT_evaluation(cfg)
                 
                means = np.mean(anomaly_results, axis=0)
                stds = np.std(anomaly_results, axis=0)
                gp_output = np.sum(means) + np.sum(0.5*stds)
                #print(anomaly_results)
            except:
                gp_output = 2.5e4
            print_vals = True
            if print_vals == True:
    
                print('%d) output: %d, h0: %.4f, maxiter: %d, a_coeff: %.4f, b_coeff: %.4f' % (counter,gp_output,cfg[0], cfg[1],cfg[2],cfg[3]))
                #print('gp_output: %.4f, h0: %.6f, maxiter: %d, a_coeff: %.6f, b_coeff: %.6f' % (gp_output, cfg[0], cfg[1],cfg[2],cfg[3]))
            
    
            return gp_output # Return the average mse plus some of its error. Biases gp_minimize to use smaller stds
        
        # GAUSSIAN PROCESS BAYESIAN OPTIMISATION TUNING PROCEDURE
        func = mesh_bayesian_evaluation
        init_time = datetime.now()
        gp_output = gp_minimize(func,                       # function to minimize
                             dimensions=bounds,             # bounds on each hyperparameter dimension 
                             acq_func="EI",                 # acquisition function (Expected improvement)
                             n_calls = num_evaluations,     # number of evaluations of func
                             n_random_starts = random_vals, # number of initial, random seeding points                       
                             random_state=3456)             # some random seed
        fin_time = datetime.now()
        print("\nBayesian GP Optimisation execution time: ", (fin_time-init_time), "(hrs:mins:secs)")
        print('Results: gp_output: %.4f, h0: %.6f, maxiter: %d, a_coeff: %.6f, b_coeff: %.6f' % (gp_output.fun,
                                      gp_output.x[0], gp_output.x[1],gp_output.x[2], gp_output.x[3]))
        hyperparams = ["gp_output","h0 internodal distance","maxiter","a_coeff node density","b_coeff node density"]
        filename = 'Bayesian_run_5'+'.csv'
        write_to_csv(filename, gp_output.func_vals, gp_output.x_iters, hyperparams, n_repeats,fin_time-init_time, cores)
        
    elif (mode==6): # Further evaluation and fine-tuning of promising mesh parameters
        counter = 0
        n_anomalies = 20
        n_repeats = 40
        #Initial 30 odd good sets of meshing parameters
        # optimal_parameter_list = [[0.052,2801,10,0.02],[0.051,2551,5.69,0.02],[0.046,2835,9.93,0.02],
        #                           [0.066,400,9.9,0.02],[0.068,1181,4.79,0.36],[0.077,1530,7.76,0.02],[0.056,1364,5.22,2.51]]
        # optimal_parameter_list = [[0.054,2953,9.17,1.86],[0.055,2624,7.65,3.85],[0.054,2873,8.21,2.48],[0.054,3000,9.34,1.89],[0.055,3000,10,5.72],
        #                           [0.054,1982,9.81,5.78],[0.068,2389,6.58,4.11],[0.053,1508,8,6],[0.054,1950,10,5.41],[0.056,2817,6.63,2.59],
        #                           [0.045,1009,3.67,3.13],[0.068,774,6.56,2.12],[0.068,905,6.98,3.25],[0.048,1540,4.64,5.88],[0.068,881,8.96,1.62],
        #                           [0.08,2197,9.86,1.6],[0.047,1426,5.25,5.99],[0.051,2415,2.69,2.72],[0.081,2695,9.55,3.14],[0.047,800,2.75,4.43]]
        
        # After the above 30 values were checked, as well as some runs on other computers, selected 4 sets which gave the lowest cumulative error
        #optimal_parameter_list = [[0.054, 3000, 9.34, 1.89], [0.054, 2953, 9.17, 1.86]]
        optimal_parameter_list = [[0.055, 2624, 7.65, 3.85],[0.045,2000, 3.67,3.13]]

        error_outputs = np.zeros(len(optimal_parameter_list))
        
        init_time = datetime.now()
        for i in range(0,len(optimal_parameter_list)): 
            params = optimal_parameter_list[i]
            error_outputs[i] = mesh_parameter_evaluation(h0=params[0], maxiter=params[1], a_coeff=params[2], b_coeff=params[3],
                                                         n_repeats=n_repeats, n_anomalies=n_anomalies)
        fin_time = datetime.now()
        
        hyperparams = ["gp_output","h0 internodal distance","maxiter","a_coeff node density","b_coeff node density"]
        filename = 'final_4_optimal_params_3'+'.csv'
        write_to_csv(filename, np.array(error_outputs), np.array(optimal_parameter_list), hyperparams, n_repeats,fin_time-init_time, cores=1)
        
    else:
        print("Script operation mode unselected or invalid")




