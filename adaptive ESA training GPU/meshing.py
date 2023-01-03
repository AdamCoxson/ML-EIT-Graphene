'''
6 April 2020

Python file that deals with basic meshing functions used to generate

forward solutions of EIT problems (built on top of pyEIT implementation

of the distmesh algorithm)

by Vasil Avramov and Ivo Mihov

in collaboration with Artem Mishchenko and Sergey Slizovskiy

from the Solid State Physics group 

at The University of Manchester

'''

import pyeit.mesh.shape as shape
import pyeit.mesh.distmesh as distmesh
import numpy as np
import matplotlib.pyplot as plt

def pfix(a=1., b=1., centre=None):
    '''

    a function that returns the corners and midpoints of edges to the respective positions

    takes:

    a for side along x - float
    b for side along y - float
    centre for position of origin - array shape (2)

    returns:

    p_fix - contains all coordinates of fixed points (corners) - array shape (4, 2)


    '''
    # initialise p_fix arrays
    
    p_fix=[]
    if centre is None:
    # if centre of sample is not set, take it to be [0, 0]
        centre=[0, 0]

    # append positions of corners to p_fix array to define square
    p_fix.append(np.array([-a/2, -b/2])-centre)
    p_fix.append(np.array([-a/2, b/2])-centre)
    p_fix.append(np.array([a/2, -b/2])-centre)
    p_fix.append(np.array([a/2, b/2])-centre)

    return np.array(p_fix)

def fix_electrodes(centre=None, edgeX=0.1, edgeY=0.1, a=1, b=1, ppl=16):
    '''
    
    function that fixes the position of electrodes and returns them together with coordinates of fixed points

    takes:

    centre for position of origin - array shape (2)
    edgeX for offset from corner (along x) - float
    edgeY for offset from corner (along y) - float
    a for side along x - float
    b for side along y - float
    ppl for number of electrodes to be fixed - int
    '''
    # sets center to be at origin if not given
    if centre is None:
        centre = [0, 0]
    # stores distance between adjacent electrodes along x and y
    delta_x = (a-2*edgeX)/(ppl/4-1);
    delta_y = (b-2*edgeY)/(ppl/4-1);
    # initialises array to store coordinates of electrodes
    el_pos = np.empty((ppl, 2), dtype=np.float64)
    ppl = int(ppl)
    # sets ppl/4 electrodes along y = +1
    el_pos[:int(np.round(ppl/4)), 0] = (centre[0] - a/2 + edgeX) + np.arange(0, ppl/4) * delta_x
    el_pos[:int(np.round(ppl/4)), 1] = (centre[1] + b/2) * np.ones(int(ppl/4))
    # sets ppl/4 electrodes along x = +1
    el_pos[int(np.round(ppl/4)):int(np.round(ppl/2)), 0] = (centre[0] + a/2) * np.ones(int(ppl/4))
    el_pos[int(np.round(ppl/4)):int(np.round(ppl/2)), 1] = (centre[1] + b/2 - edgeY) - np.arange(0, ppl/4) * delta_y
    # sets ppl/4 electrodes along y = -1
    el_pos[int(np.round(ppl/2)):int(np.round(3*ppl/4)), 0] = (centre[0] + a/2 - edgeX) - np.arange(0, ppl/4) * delta_x
    el_pos[int(np.round(ppl/2)):int(np.round(3*ppl/4)), 1] = (centre[1] - b/2) * np.ones(int(ppl/4))
    # sets ppl/4 electrodes along x = -1
    el_pos[int(np.round(3*ppl/4)):ppl, 0] = (centre[0] - a/2) * np.ones(int(ppl/4))
    el_pos[int(np.round(3*ppl/4)):ppl, 1] = (centre[1] - b/2 + edgeY) + np.arange(0, ppl/4) * delta_y
    # returns two arrays, first for corner positions, second for electrode positions
    return pfix(a, b, centre), el_pos

def fix_electrodes_multiple_odd(centre=None, edgeX=0.1, edgeY=0.1, a=1, b=1, ppl=20, w=0.2, num_per_el=1):
    '''
    
    function that fixes the position of electrodes and returns them together with coordinates of fixed points

    takes:

    centre for position of origin - array shape (2)
    edgeX for offset from corner (along x) - float
    edgeY for offset from corner (along y) - float
    a for side along x - float
    b for side along y - float
    ppl for number of electrodes to be fixed - int
    w for width of each electrode - float
    num_per_el for number of nodes per electrode minus 1 divided by 2 (n = num_per_el*2 + 1) - int
    
    returns:

    '''
    # sets center to be at origin if not given
    if centre is None:
        centre = [0, 0]
    # stores distance between adjacent electrodes along x and y
    delta_x = (a-2*edgeX)/(ppl/4-1);
    delta_y = (b-2*edgeY)/(ppl/4-1);
    # initialises array to store coordinates of electrodes
    el_pos = np.empty((ppl * (2*num_per_el + 1), 2), dtype=np.float64)
    ppl = int(ppl)
    if num_per_el == 0:
        w = 0
        n_s = 1
    else:
        n_s = num_per_el
    i = np.arange(-num_per_el, num_per_el+1)
    i = np.tile(i, (1, int(ppl/4))).T.ravel()
    arr_prop = np.tile(np.arange(0, ppl/4), (2*num_per_el+1, 1)).T.ravel()
    # sets ppl/4 electrodes along y = +1
    el_pos[:int((2*num_per_el+1) * np.round(ppl/4)), 0] = (centre[0] - a/2 + edgeX + i * w / (2*n_s)) + arr_prop * delta_x
    el_pos[:int((2*num_per_el+1) * np.round(ppl/4)), 1] = (centre[1] + b/2) * np.ones(int((2*num_per_el + 1) * ppl/4))
    # sets ppl/4 electrodes along x = +1
    el_pos[int((2*num_per_el+1) * np.round(ppl/4)):int((2*num_per_el+1) * np.round(ppl/2)), 0] = (centre[0] + a/2) * np.ones(int((2*num_per_el + 1) * ppl/4))
    el_pos[int((2*num_per_el+1) * np.round(ppl/4)):int((2*num_per_el+1) * np.round(ppl/2)), 1] = (centre[1] + b/2 - edgeY - i * w / (2*n_s)) - arr_prop * delta_y
    # sets ppl/4 electrodes along y = -1
    el_pos[int((2*num_per_el+1) * np.round(ppl/2)):int((2*num_per_el+1) * np.round(3*ppl/4)), 0] = (centre[0] + a/2 - edgeX - i * w / (2*n_s)) - arr_prop * delta_x
    el_pos[int((2*num_per_el+1) * np.round(ppl/2)):int((2*num_per_el+1) * np.round(3*ppl/4)), 1] = (centre[1] - b/2) * np.ones(int((2*num_per_el + 1) * ppl/4))
    # sets ppl/4 electrodes along x = -1
    el_pos[int((2*num_per_el+1) * np.round(3*ppl/4)):(2*num_per_el + 1) * ppl, 0] = (centre[0] - a/2) * np.ones(int((2*num_per_el + 1) * ppl/4))
    el_pos[int((2*num_per_el+1) * np.round(3*ppl/4)):(2*num_per_el + 1) * ppl, 1] = (centre[1] - b/2 + edgeY + i * w / (2*n_s)) + arr_prop * delta_y

    return pfix(a, b, centre), el_pos

def fix_electrodes_multiple(centre=None, edgeX=0.1, edgeY=0.1, a=2, b=2, ppl=16, el_width=0.1, num_per_el=None, start_pos='left'):
    '''
    
    function that fixes the position of electrodes and returns them together with coordinates of fixed points

    takes:

    centre for position of origin - array shape (2)
    edgeX for offset from corner (along x) - float
    edgeY for offset from corner (along y) - float
    a for side along x - float
    b for side along y - float
    ppl for number of electrodes to be fixed - int
    el_width for width of each electrode - float
    num_per_el for number of nodes per electrode - int
    start_pos defines whether the electrode number starts from the left, centre or right of an edge and continues clockwise.
    
    returns:
    
    array pfix of positions of fixed points for rectangle
    array el_pos with positions of nodes of electrodes

    '''

    if centre is None:
        centre = [0, 0]

    if num_per_el is None:
        num_per_el = 1
    
    delta_x = (a - 2. * edgeX - (ppl/4 * el_width) )/(ppl/4-1) + el_width;
    delta_y = (b - 2. * edgeY - (ppl/4 * el_width) )/(ppl/4-1) + el_width;
    
    el_pos = np.empty((ppl*num_per_el, 2), dtype=np.float64)
    ppl = int(ppl)

    dist_el = np.arange(num_per_el) * el_width / (num_per_el - 1)
    dist_arr = ((np.arange(0, ppl/4) * delta_x)[:, None] + dist_el [None, :]).ravel()
    
    el_pos[:int(np.round(ppl/4 * num_per_el)), 0] = (centre[0] - a/2 + edgeX) + dist_arr
    el_pos[:int(np.round(ppl/4 *num_per_el )), 1] = (centre[1] + b/2) * np.ones(int(ppl/4*num_per_el))
    
    el_pos[int(np.round(ppl/4 *num_per_el)):int(np.round(ppl/2 *num_per_el)), 0] = (centre[0] + a/2) * np.ones(int(ppl/4*num_per_el))
    el_pos[int(np.round(ppl/4 *num_per_el)):int(np.round(ppl/2 *num_per_el)), 1] = (centre[1] + b/2 - edgeY) - dist_arr
    
    el_pos[int(np.round(ppl/2 *num_per_el)):int(np.round(3*ppl/4*num_per_el)), 0] = (centre[0] + a/2 - edgeX) - dist_arr
    el_pos[int(np.round(ppl/2*num_per_el)):int(np.round(3*ppl/4*num_per_el)), 1] = (centre[1] - b/2) * np.ones(int(ppl/4*num_per_el))
    
    el_pos[int(np.round(3*ppl/4*num_per_el)):ppl*num_per_el, 0] = (centre[0] - a/2) * np.ones(int(ppl/4*num_per_el))
    el_pos[int(np.round(3*ppl/4*num_per_el)):ppl*num_per_el, 1] = (centre[1] - b/2 + edgeY) + dist_arr
    
    if (start_pos=='mid'):
        shift = (ppl//(4*2))*num_per_el # This ensures electrode numbering start point is at the centre of one side
        el_pos = np.roll(el_pos, shift, axis=0) # Shift to 9 o'clock position
    
    return pfix(a, b, centre), el_pos

def min_dist_to_el(pts, el_pos):
    '''
    
    function that finds minimum position from array of points to closest electrode

    takes:

    pts for array of node coordinates of mesh - array shape (len(pts), 2)
    el_pos for array of electrode coordinates - array shape (number_electrodes, 2)

    returns:

    array with distances to closest electrode - array shape (len(pts))

    '''
    # find distance of every point to each electrode (euclidean)
    pts = np.array(pts)
    el_pos = np.array(el_pos)
    d = np.sqrt(np.power(pts[None, :, 0] - el_pos[:, None, 0], 2) + np.power(pts[None, :, 1] - el_pos[:, None, 1], 2))
    return np.min(d, axis=0)#cp.asnumpy(cp.min(d, axis=0))


def mesh_gen(n_el=20, num_per_el=3, start_pos='left', el_width=0.1, side_length=2.0, edge=0.1, mesh_params=None):
    ''' 

    function that generates the mesh to be used in solving the FEM and inverse problem

    takes:

    n_el - number of electrodes - int (=20 by default)

    returns:

    mesh_dict - dictionary of mesh characteristics:
        p - array of mesh node coordinates - array shape (number of nodes, 2)
        t - indices of triangle vertices in p array (counter-clockwise) - array shape (number of triangles, 3)
        el_pos - positions of electrodes on plate (clockwise) - array shape (n_el, 2)
        p_fix - coords of fixed points - array shape (number of fixed points, 2)

    '''
    if mesh_params == None:
        mesh_params = [0.1, 1200, 0.45, 0.2]
    h0      = mesh_params[0]
    maxiter = mesh_params[1]
    a_coeff = mesh_params[2]
    b_coeff = mesh_params[3]
    
    edgeX = edge
    edgeY = edge
    a = side_length
    b = side_length
    # defining the shape of the plate (a rectangle of sides a and b)
    def _fd(pts):
        return shape.rectangle(pts, p1=[-1., -1.], p2=[1., 1.])
    # (optional) variable meshing with smaller triangles close to the electrodes - taken by the DISTMESH class (pyEIT)
    def _fh(pts):
        p_fix, el_pos = fix_electrodes_multiple(edgeX=edgeX, edgeY=edgeY, a=a, b=b, ppl=n_el,
                                                     el_width=el_width, num_per_el=num_per_el, start_pos=start_pos)
        # if (num_per_el!=1):
        #     p_fix, el_pos = fix_electrodes_multiple(edgeX=edgeX, edgeY=edgeY, a=a, b=b, ppl=n_el,
        #                                             el_width=el_width, num_per_el=num_per_el, start_pos=start_pos)
        # elif(num_per_el==1):
        #     p_fix, el_pos = fix_electrodes(edgeX=edge, edgeY=edge, a=a, b=b, ppl=n_el)
        
        #p_fix, el_pos = fix_points_rectangle(edgeX=0.2, edgeY=0.2, a=2, b=2)
        minDist = min_dist_to_el(pts, el_pos)
        #return 5.0 + 3.0 * np.power(minDist/np.amax(minDist), 2)
        return a_coeff + b_coeff* np.power(minDist/np.amax(minDist), 2)
    # setting the electrodes and fixed points positions in p_fix array
    # if (num_per_el!=1):
    #         p_fix1, el_pos = fix_electrodes_multiple(edgeX=edge, edgeY=edge, a=a, b=b, ppl=n_el, el_width=el_width,
    #                                                  num_per_el=num_per_el, start_pos=start_pos)
    # elif(num_per_el==1):
    #         p_fix1, el_pos = fix_electrodes(edgeX=edge, edgeY=edge, a=a, b=b, ppl=n_el)
    p_fix1, el_pos = fix_electrodes_multiple(edgeX=edgeX, edgeY=edgeY, a=a, b=b, ppl=n_el,
                                                     el_width=el_width, num_per_el=num_per_el, start_pos=start_pos)
    p_fix = np.empty((len(el_pos)+len(p_fix1), 2), dtype = 'f4')
    p_fix[0:len(el_pos)] = el_pos
    p_fix[len(el_pos):len(el_pos)+len(p_fix1)] = p_fix1
    # building the DISTMESH class object (from pyEIT)
    #p, t = distmesh.build(_fd, _fh, pfix=p_fix, h0=0.08, maxiter=2000)
    p, t = distmesh.build(_fd, _fh, pfix=p_fix, bbox=None, h0=h0,
          densityctrlfreq=32, deltat=0.2,
          maxiter=maxiter, verbose=False)
    #p, t = distmesh.build(_fd, _fh, pfix=p_fix, h0=0.08, maxiter=1500)
    # creating the dictionary with the specifics of the meshing
    mesh_dict = {'p': p , 't' : t, 'el_pos' : el_pos, 'p_fix': p_fix}
    

    do_plot = False
    if do_plot == True:
        fig, ax = plt.subplots()
        ax.triplot(p[:, 0], p[:, 1], t)
        #ax.plot(p_fix[:, 0], p_fix[:, 1], 'ro')
        ax.set_aspect('equal')
        #ax.set_xlabel(c0)
        #ax.set_ylabel(c1)
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        plt.yticks(ticks=[-1.0,-0.50,0,0.50, 1.00], labels=[-1.0,-0.50,0,0.50,1.00], fontsize=12)
        plt.xticks(ticks=[-1.0,-0.50,0,0.50, 1.00], labels=[-1.0,-0.50,0,0.50,1.00], fontsize=12)
        ax.plot(el_pos[:, 0], el_pos[:, 1], 's', markersize = 2.0, color = 'red')
        plt.show()
    
    # return dictionary
    return mesh_dict

def mesh(n_el=20, num_per_el=3, start_pos='left', el_width=0.1, side_length=2.0, edge=0.1, mesh_params=None):
    # generate and extract meshing
    mesh_dict = mesh_gen(n_el=n_el, num_per_el=num_per_el, start_pos=start_pos,
                         el_width=el_width, side_length=side_length,edge=edge, mesh_params=mesh_params)
    p = mesh_dict['p']
    t = mesh_dict['t']
    del mesh_dict
    # create a constant permitivity array for reference
    perm = np.ones(t.shape[0], dtype=np.float)
    t = checkOrder(p, t) # check that order of vertices in each triangle is counter-clockwise
    mesh_obj = {'element': t,
                'node':    p,
                'perm':    perm}
    return mesh_obj





def checkOrder(p, t):
    '''
    function that checks if vertices of each triangle are in counter-
    clockwise order by checking if area (determinant) is positive 
    and reorders otherwise

    takes:

    p - array with coordinates of all mesh nodes - shape (num_nodes, 2)
    t - array with all indices (in p) of triangle vertices - shape (num_triangles, 3)
    
    returns:

    t - reordered array with all indices (in p) of triangle vertices - shape (num_triangles, 3)
    '''
    xy = p[t]
    s = xy[:, [2, 0, 1]] - xy[:, [1, 2, 0]]
    s1 = s[:, 0]
    s2 = s[:, 1]
    area = 0.5 * (s1[:, 0]*s2[:, 1] - s1[:, 1]*s2[:, 0])
    ind = area < 0
    try:
        t[ind] = t[ind, [0, 2, 1]]
    except:
        pass
    return t