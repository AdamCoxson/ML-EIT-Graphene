from anomalies import *
from meshing import *
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import numpy.random as rand

import cupy as cp

from collections import namedtuple

from pyeit.mesh import set_perm
from pyeit.eit.fem import Forward
import pyeit.eit.greit as greit

import skimage.util as sk

from time import time

def generateExMat(ne=20, el_dist=None):
    #generates all possible excitation pairs
    if el_dist is None:
        el_dist = np.arange(ne)
    A = np.arange(ne)
    M = np.dot(np.ones(ne)[:,None], A[None, :]) % ne
    N = (M + el_dist[:, None]) % ne    
    ex_mat = np.stack((N, M), axis=-1)
    ex_mat = ex_mat.reshape(((ne)**2, 2))

    ind = ex_mat[:,0] < ex_mat[:, 1]

    ex_mat = ex_mat[ind]

    #print(ex_mat.shape)

    return ex_mat.astype(np.int32)

def orderedExMat(n_el=20, length=None, el_dist=None):
    ''' 
    function that generates ex_mat with an order that provides more information about the conductivity
    first measure with the electrodes closest to edge (0, 5, 10, 15) (% 5 = 0), then with (1, 6, 11, 16) (% 5 = 1) and so on
    takes:

    n_el - number of electrodes - int
    length - length of excitation matrix ( raises exception when any of el_dist is 10) - int
    el_dist - scalar or array that give distance between source and sink for each line of excitation matrix - int or array (get treated differently)
    
    returns:

    ex_mat - excitation matrix of sources and sinks that are used to reconstruct permittivity - array shape (num_pairs, 2)

    '''
    ex_mat_all = generateExMat(n_el)
    if length is None:
        length = rand.randint(7, 26)
    if np.isscalar(el_dist):
        el_dist = int(el_dist) * np.ones(n_el).astype('i2')
    elif el_dist is None:
        a = np.arange(1, n_el).tolist()
        a.append('random')
        split = rand.choice(np.arange(len(a)))
        if a[split] == 'random':
            el_dist = rand.randint(1, n_el, length)
            NA, limit = np.unique(el_dist, return_counts=True)
            while limit.any()>n_el:
                el_dist = rand.randint(1, n_el, length)
                NA, limit = np.unique(el_dist, return_counts=True)
            #print('random')
        else:
            el_dist_main = rand.randint(1, n_el, length//n_el)
            el_dist_main = np.tile(el_dist_main[:, None], [1,n_el]).reshape((n_el * (length//n_el), 1))
            if length//n_el != 0:
                ind = np.arange(1, n_el)[~(np.amax(el_dist_main==np.arange(1, n_el)[None], axis=0))]
            else:
                ind = np.arange(1, n_el)
            print("a:",a)
            print("ind: ", ind)
            el_dist_rem = np.ones(shape=(length%n_el, 1)) * np.random.choice((ind))
            el_dist = np.vstack((el_dist_main, el_dist_rem)).astype('i2')
    else:
        raise TypeError('el_dist must be array or scalar')
    #print(el_dist)
    a, c = np.unique(el_dist, return_counts=True)
    if (c>n_el-a).any():
        d = np.zeros(n_el - 1)
        d[a-1] = c
        ind = np.where(c > n_el - a)
        d[n_el - a[ind] - 1] = d[n_el - a[ind] - 1] + d[a[ind] - 1] - (n_el - a[ind])
        d[a[ind] - 1] = n_el - a[ind]
        i = n_el - 1 - np.arange(len(d))
        d[d > i] = i[d > i]
        c = d[d != 0].astype(int)
        a = np.nonzero(d)[0] + 1
    distIndex = np.diff(ex_mat_all, axis=1)
    indices = np.where(distIndex[:, None] == a[None])

    assert len(indices[0]) == np.sum(n_el - a)
    sumA = np.cumsum(n_el - a)
    indA = np.empty(len(sumA))
    indA[1:] = sumA[:-1]
    indA[0] = 0
    indA = indA.astype(int)
    ind=[]
    for i in range(len(a)):
        shuff = np.arange((n_el - a[i]))
        shuff = shuff[np.argsort(shuff % 5)]
        ind.append(indices[0][indA[i]+shuff[:c[i]]])

    ind = np.hstack(ind)
    ex_mat = ex_mat_all[ind]
    return ex_mat.astype(np.int32)


def excitation(n_el, length = None, el_dist = None):
    '''
    function that randomises excitation matrix using 3 distance choices (in terms of index)
    choices are: dist = 1, dist = n_el/2 (opposite measurements), dist = n_el/4 (pi/2 measurements)

    takes: 

    n_el - number of electrodes - int

    returns:

    ex_mat - excitation array - shape (n_el, 2)

    '''
    
    el_dist = np.array([1 / n_el, 1 / 2, 1 / 4]) * n_el
    #initialise arrays for each option
    ex_mat = np.empty((n_el, 2), dtype='i2')
    ex_mat_1 = np.empty((n_el, 2), dtype='i2')
    ex_mat_1_2 = np.empty((n_el, 2), dtype='i2')
    ex_mat_1_4 = np.empty((n_el, 2), dtype='i2')
    #fill the arrays with measurement pairs
    ex_mat_1[:] = np.vstack(((np.arange(n_el)),
                        ((el_dist[0] + np.arange(n_el)))%n_el)).T
    ex_mat_1_2[:] = np.vstack(((np.arange(n_el)),
                        ((el_dist[1] + np.arange(n_el)))%n_el)).T
    ex_mat_1_4[:] = np.vstack(((np.arange(n_el)),
                        ((el_dist[2] + np.arange(n_el)))%n_el)).T
    #randomise from which ex_mat should pairs be taken
    rand_i = rand.choice(np.arange(3), size=n_el)
    #create choice array from which all measurement pairs are chosen
    ex_choice = np.array([ex_mat_1, ex_mat_1_2, ex_mat_1_4])
    #initialise indices to be used, when generating final exictation matrix
    ind = np.vstack((rand_i, np.arange(n_el))).T
    #intialise randomised excitation matrix, given the random choices already taken
    ex_mat[:, :] = ex_choice[ind[:, 0], ind[:, 1], :]
    return ex_mat
    
    #ex_mat_all = generateExMat(n_el)
    '''print(ex_mat_all)
    if length is None:
        length = rand.randint(6, 26)
    
        if a[split] == 'random':
            el_dist = rand.randint(1, 19, length)
            #el_main = np.
        else:
            if length <= n_el:
                el_dist = rand.randint(1, 19, length)
            if length > n_el:
                el_dist = rand.randint(1, 19, length//n_el + 1)
    else:
        if len(el_dist) < length//n_el + 1:
            raise ValueError('Not enough el_dist for given length')
        else:
            pass
            '''

    #el_dist = np.diff(ex_mat_all, axis=1)
    '''
    a = np.ones(17)
    split = 0
    if length is None:
        length = rand.randint(7, 26)
    
    if np.isscalar(el_dist):
        el_dist *= np.ones(length)
    

    #el_dist = np.array(el_dist)
    if el_dist is None:
        a = np.arange(1, 19).tolist()
        a.append('random')
        split = rand.choice(np.arange(len(a)))
        if a[split] == 'random':
            el_dist = rand.randint(1, 19, length)
        elif length < n_el:
            el_dist = rand.randint(1, 19)
        elif length % n_el == 0:
            el_dist = rand.randint(1, 19, length//n_el)
        elif length > n_el:
            el_dist = rand.randint(1, 19, length//n_el + 1)

    if length != n_el:

        if a[split] == 'random':
            assert len(el_dist) == length
            if length > 20:
                num_index = np.argsort(np.arange(length) % 5)
                choice_arr = (np.arange(length)[num_index])%n_el
                ex_mat = np.concatenate((col_0[:, None], col_1[:, None]), axis = 1)
            else:
                num_index = np.argsort(np.arange(n_el) % 5)
                choice_arr = (np.arange(n_el)[num_index])%n_el [:length]
                col_0 = (choice_arr).reshape(length, 1)
                col_1 = (np.arange(length)%n_el).reshape(length, 1)
                ex_mat = np.concatenate((col_0, col_1), axis = 1)

        else:
            if length > 20:
                assert len(el_dist) == (length//n_el + 1)
                el_dist_temp = np.array(el_dist)
                el_dist_temp = np.tile(el_dist_temp[:, None], [1, n_el])
                ex_mat_temp = np.tile(np.arange(n_el), [1, len(el_dist_temp)])

                ex_mat_temp2 = (ex_mat_temp + el_dist_temp) % n_el

                ex_mat = np.concatenate((ex_mat_temp.ravel()[:, None], ex_mat_temp2.ravel()[:, None]), axis=1)

                el_dist_rem = (rand.randint(1, 19)) * np.ones(length % n_el)
                num_index = np.argsort(np.arange(n_el) % 5)[:(length % n_el)]

                ex_mat_rem1 = np.arange(n_el)[num_index]
                ex_mat_rem2 = (np.arange(n_el)[num_index] + el_dist_rem) % n_el
                ex_mat_rem = np.concatenate((ex_mat_rem1[:, None], ex_mat_rem2[:, None]), axis=1)
                ex_mat = np.vstack((ex_mat, ex_mat_rem))
            else:
                assert np.isscalar(el_dist)
                el_dist = rand.randint(1, 19) * np.ones(length)
                num_index = np.argsort(np.arange(n_el) % 5)
                choice_arr = (np.arange(n_el)[num_index])[:length]
                col_2 = (choice_arr + el_dist) % n_el
                ex_mat = np.concatenate((choice_arr[:, None], col_2[:, None]), axis=1)
    else:
        assert np.isscalar(el_dist)
        col_1 = np.arange(n_el)
        col_2 = (np.arange(n_el) + el_dist) % n_el

        ex_mat = np.concatenate((col_1[:, None], col_2[:, None]), axis=1)

    print(ex_mat)
    index_XM = np.sum(np.equal(ex_mat[:, None], ex_mat[:, [1,0]][None]), axis=1) == 2
    indices = np.where(index_XM)
    ind = (indices[0] > indices[1])
    indices = [indices[0][ind], indices[1][ind]]
    i = np.ones(len(ex_mat), dtype='i4')
    indices = np.unique(indices[0])
    i[indices] = 0
    i= i.astype(bool)
    ex_mat = ex_mat[i]






    el_dist = rand.randint(1, 20, (length))

    if np.isscalar(el_dist):
        el_dist *= np.ones((length))

    ex_mat_all = generateExMat(n_el)

    distIndex = np.diff(ex_mat_all, axis=1)
    distIndex[distIndex<0] += 20
    indices = np.where(distIndex[:, None] == el_dist[None])
    A = ex_mat_all[indices[0]]

    return ex_mat

    '''


def greit_rec(p, t, el_pos, anomaly, fwd, continuous=False, step='random', n_pix=64, n_el=20, length=None, el_dist=None, num_per_el=3, continuousPerm=None):
    '''
    function that generates forward solution and a GREIT reconstruction for a given conductivity distribution
    
    Takes:
    p - array of point coordinates created by meshing algorithm [number of points, 2] array[float]
    t - array of a triangles included in meshing [number of triangles , 3] array[int]
    el_pos - array of electrode positions [number of electrodes , 2] array[float]
    anomaly - list of all the anomalies for which reconstruction should be made array[dict]
    step - array storing the distance (in number of electrodes) for each source/sink pair [number of source/sink pairs , 1] array[int]
    (Default is 'random' == generates random step for each measurement)
    n_pix - number of pixels to be created in each dimension for GREIT algorithm [int]
    n_el - number of electrodes of the system that participate in reconstruction [int]

    Returns:
    ds - recreated conductivity map by GREIT algorithm [number of pixels,number of pixels] array[float]
    '''
    #check order of points for each triangle and rearrange to ensure counter-clockwise arrangement (from pyEIT)
    t = checkOrder(p, t)
    #generate electrode indices
    el_pos = np.arange(n_el * num_per_el)
    #initialise unit uniform permitivity for the triangular mesh to be used
    perm = np.ones(t.shape[0], dtype=np.float)
    #build mesh structure given the input elements
    mesh_obj = {'element': t,
                'node': p,
                'perm': perm}

    # extract x, y coordinate of mesh vertices
    x, y = p[:, 0], p[:, 1]
    #check to see the state of the generated mesh (optional)
    #quality.stats(p, t)

    ex_mat = orderedExMat(n_el, length, el_dist)
    #if random step is invoked a random step is generated for each source/sink pair
    if step == 'random':
        step = np.array(rand.randint(1, n_el, size=ex_mat.shape[0]))
    #if the step is not an integer in the needed range, just randomize the same step for a source/sink pairs
    elif (step < 0 or step > n_el).any():
        step = rand.randint(1, n_el)
    start_t = time()
    # calculate simulated data using FEM (for uniform conductivity)
    f = fwd.solve_eit(ex_mat=ex_mat, step=step, perm=mesh_obj['perm'])
    # creating forward dictionary with solution of forward problem
    pde_result = namedtuple("pde_result", ['jac', 'v', 'b_matrix'])
    f0 = pde_result(jac=f.jac,
                    v=f.v,
                    b_matrix=f.b_matrix)
    # adding anomalies with a overwritten anomaly function originally in pyEIT (pyEIT.mesh.wrapper) (heavily changed)
    unp_t = time()
    print("Unperturbed forward solution t", unp_t - start_t)
    if continuous == False:
        mesh_new = set_perm(mesh_obj, anomaly=anomaly, background=None)
        permUsed = mesh_new['perm']
    elif continuous == True:
        permUsed = continuousPerm
    else:
        print('kurec')
    start_anom = time()
    # solving with anomalies
    f1 = fwd.solve_eit(ex_mat=ex_mat, step=step, perm=permUsed)
    # generate array for variance, 3% standard deviation for the voltage measurement
    variance = 0.0009 * np.power(f1.v, 2)
    # apply noise to voltage measurements v
    # here white Gaussian noise is assumed (since we don't have significant sources of systematic error like in medical applications, e.g. moving electrodes...)
    v = sk.random_noise(f1.v, 
                        mode='gaussian', 
                        clip=False, 
                    	mean=0.0, 
                        var=variance)
    # create the Forward object used by GREIT the voltage map with the Gaussian noise
    f1 = pde_result(jac=np.vstack(f1.jac),
                    v=np.hstack(v),
                    b_matrix=np.vstack(f1.b_matrix))
    end_anom = time()
    print("Anomaly forward t: ", end_anom - start_anom )
    # (optional) draw anomalies only
    
    do_plot = 1
    if (do_plot==1):
        delta_perm = np.real(mesh_new['perm'] - mesh_obj['perm'])
        fig, ax = plt.subplots()
        im = ax.tripcolor(p[:, 0], p[:, 1], t, delta_perm,
                          shading='flat', cmap=plt.cm.viridis)
        ax.set_title(r'$\Delta$ Conductivities')
        fig.colorbar(im)
        ax.axis('equal')
        fig.set_size_inches(6, 4)
        # fig.savefig('demo_bp_0.png', dpi=96)
        #plt.show()
    
    start_greit = time()
    # constructing GREIT object (from pyEIT) (classes changed singificantly for optimisation)
    eit = greit.GREIT(mesh_obj, el_pos, f=f, ex_mat=ex_mat, step=step, parser='std')
    # solving inverse problem with GREIT algorithm
    eit.setup(p=0.1, lamb=0.01, n=n_pix, s=15., ratio=0.1)
    ds = eit.solve(f1.v, f0.v)
    #reshaping output to the desired dimensions
    ds = ds.reshape((n_pix, n_pix))
    print("Greit solution time ", time() - start_greit)
    # (optional) plot to check whether generated sensibly
    if (do_plot==1):
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(np.real(ds), interpolation='none', cmap=plt.cm.viridis, origin='lower', extent=[-1,1,-1,1])
        fig.colorbar(im)
        ax.axis('equal')
        
        plt.show()
    '''
    gradConductivity = np.linalg.norm(np.gradient(ds), axis=0)
    figGrad, axGrad = plt.subplots(figsize=(6, 4))
    imGrad = axGrad.imshow(np.real(gradConductivity), interpolation='none', cmap=plt.cm.viridis, origin='lower', extent=[-1,1,-1,1])
    figGrad.colorbar(imGrad)
    axGrad.axis('equal')

    figGrad2, axGrad2 = plt.subplots(figsize=(6, 4))
    imGrad2 = axGrad2.imshow(np.real(gradConductivity * ds), interpolation='none', cmap=plt.cm.viridis, origin='lower', extent=[-1,1,-1,1])
    figGrad2.colorbar(imGrad2)
    axGrad2.axis('equal')


    v_pert = np.empty(shape=(len(f1.v), len(f1.v)))
    perturbing_mat = np.ones((len(f1.v), len(f1.v))) + 0.05 * np.identity(len(f1.v))
    v_pert[:] = np.dot(perturbing_mat, np.diag(f1.v))
    influence_mat = -np.dot(eit.H, v_pert).reshape(n_pix, n_pix, len(f1.v)) - ds[:, :, None]
    influence_mat = np.absolute(influence_mat)
    influence_mat = np.sum(influence_mat, axis=2)
    
    #mask = circleMask(npix, a)
    #influence_mat[~mask] = np.amax(influence_mat)

    figInfl, axInfl = plt.subplots(figsize=(6, 4))
    imInfl = axInfl.imshow(np.real(influence_mat), interpolation='none', cmap=plt.cm.viridis, origin='lower', extent=[-1,1,-1,1])
    figInfl.colorbar(imInfl)
    axInfl.axis('equal')

    totalMap = gradConductivity * ds * influence_mat
    figTotal, axTotal = plt.subplots(figsize=(6, 4))
    imTotal = axTotal.imshow(np.real(totalMap), interpolation='none', cmap=plt.cm.viridis, origin='lower', extent=[-1,1,-1,1])
    figTotal.colorbar(imTotal)
    axTotal.axis('equal')
    plt.show()
    '''
    return ds

