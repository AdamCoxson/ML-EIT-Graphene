'''
6 April 2020

Python file for generating anomalies in sample conductivity distributions.

by Ivo Mihov and Vasil Avramov 

in collaboration with Artem Mischenko and Sergey Slizovskiy 

from Solid State Physics Group 

at the University of Manchester

'''

import numpy as np
#import numpy.random as rand
from random import SystemRandom
import numpy.linalg as la
rand = SystemRandom()
def multivariateGaussian(x, mu, sigma, normalised=False):
    if normalised:
        denominator = 1. / (2 * np.pi * np.sqrt(np.linalg.det(sigma)))
    else:
        denominator = 1.
    x_centred = x - mu
    #path = np.einsum_path('ij, jk, ki->i', x_centred, np.linalg.inv(sigma), x_centred.T, optimize='optimal')[0]
    numerator = np.exp(-0.5 * np.einsum('ij, jk, ki->i', x_centred, np.linalg.inv(sigma), x_centred.T, optimize='optimal'))
    return numerator / denominator

def generateContinuousConductivity(a, centre, number_of_gaussians, mu, sigma, npix, weightGauss=None, pts=None, tri=None):
    # array to store permitivity in different square
    if centre is None:
        centre=[0, 0]
    if (number_of_gaussians) == 0:
        if pts is not None and tri is not None:
            return np.ones((npix, npix)), np.ones(tri.shape[0])
        else:
            return np.ones((npix, npix)), None
    # assumed that background permitivity is 1 (and therefore difference with uniform will be 0)
    permSquares = np.zeros((int(npix), int(npix)), dtype='f4')
    if pts is not None and tri is not None:
        permTri = np.zeros(tri.shape[0], dtype='f4')
    # initialises an array to store the coordinates of centres of squares (pixels)
    centresSquares = np.empty((npix, npix, 2), dtype='f4')
    # initialising the j vector to prepare for tiling
    j = np.arange(npix)
    # tiling j to itself npix times (makes array shape (npix, npix))
    j = np.tile(j, (npix, 1))
    # i and j are transposes of each other    
    i = j
    j = j.T
    # assigning values to C_sq 
    centresSquares[i, j, :] = np.transpose([a / 2 * ((2 * i + 1) / npix - 1) + centre[0], a / 2 * ((2 * j + 1) / npix - 1) + centre[1]])
    if pts is not None and tri is not None:
        centresTriangles = np.mean(pts[tri], axis=1)
    centresSquares = centresSquares.reshape((npix * npix, 2))
    if weightGauss is None:
        weightGauss = rand.uniform(size=(number_of_gaussians,), low=0., high=0.1)
    for i in range(number_of_gaussians):
        if type(weightGauss) is np.ndarray:
            weight = weightGauss[i]
        elif type(weightGauss) is float:
            weight = weightGauss
        else:
            raise TypeError("weight is not float or array of floats")
        permSquares[:] += (weight/number_of_gaussians) * multivariateGaussian(centresSquares, mu[i], sigma[i]).reshape(npix, npix)
        if pts is not None and tri is not None:
            permTri[:] += (weight/number_of_gaussians) * multivariateGaussian(centresTriangles, mu[i], sigma[i])
    if pts is not None and tri is not None:
        if (np.abs(permSquares) < 5e-2).any():
            a = np.random.randint(low = 4, high = 14) * 0.1
            permSquares += a
            permTri += a
    '''
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(np.real(permSquares) - np.ones((npix, npix)), interpolation='none', cmap=plt.cm.viridis, origin='lower', extent=[-1,1,-1,1])
    fig.colorbar(im)
    ax.axis('equal')
    #plt.show
    '''
    if pts is not None and tri is not None:
        return permSquares, permTri
    else:
        return permSquares, None

def randomiseGaussianParams(a, centre, npix):
    if centre is None:
        centre = [0, 0]
    # randomise parameters of gaussians
    number_of_gaussians = rand.randint(low=0, high=5)
    if (number_of_gaussians) == 0:
        return 0, 0, 0
    mu = np.empty(shape=(number_of_gaussians, 2))
    mu[:, 0] = rand.normal(size=(number_of_gaussians), loc=0, scale=0.5)
    mu[:, 1] = rand.normal(size=(number_of_gaussians), loc=0, scale=0.5)

    sigma = np.empty(shape=(number_of_gaussians, 2, 2))
    sigma[:, 0, 0] = rand.uniform(size=(number_of_gaussians), low = 0.2, high = 5.)
    sigma[:, 1, 1] = rand.uniform(size=(number_of_gaussians), low = 0.2, high = 5.)
    sigma[:, 1, 0] = rand.uniform(size=(number_of_gaussians), low = -np.sqrt(sigma[:,0,0]*sigma[:,1,1]), high = np.sqrt(sigma[:,0,0]*sigma[:,1,1]))
    sigma[:, 0, 1] = sigma[:, 1, 0]

    return number_of_gaussians, mu, sigma

def randomiseGaussianParam(a=2., centre=None, npix=64):
    if centre is None:
        centre = [0, 0]
    # randomise parameters of gaussians
    
    mu = np.empty(shape=(1, 2))
    mu[:, 0] = rand.uniform(-1, 1)
    mu[:, 1] = rand.uniform(-1, 1)
    #mu[:, 0] = rand.normal(size=(number_of_gaussians), loc=0, scale=0.5)
    #mu[:, 1] = rand.normal(size=(number_of_gaussians), loc=0, scale=0.5)

    sigma = np.empty(shape=(1, 2, 2))
    sigma[:, 0, 0] = rand.uniform(0.08, 0.8)
    sigma[:, 1, 1] = rand.uniform(0.08, 0.8)
    sigma[:, 1, 0] = rand.uniform(-np.sqrt(sigma[:,0,0]*sigma[:,1,1]), np.sqrt(sigma[:,0,0]*sigma[:,1,1]))
    sigma[:, 0, 1] = sigma[:, 1, 0]

    return mu, sigma

def triangle_area(x, y):
    '''
    function that area given 2d coordinates of all vertices of triangle

    takes:

    x - array storing the x-coordinates of all vertices [3, 1] float
    y - array storing the y-coordinates of all vertices [3, 1] float

    returns:

    area of the triangle
    '''
    return 0.5 * np.absolute(x[0] * (y[1] - y[2]) + x[1] * (y[2] - y[0]) + x[2] * (y[0] - y[1]))

def generate_examplary_output(a, npix, anomaly, centre=None):
    '''
    a function that generates true conductivity map to be used in training of U-net

    takes: 

    a - side of square - float
    npix - number of pixels along each axis - int
    anomaly - dictionary of anomalies characteristics
    centre - centre of coordinate system - array shape (2,)
    
    returns:

    true conductivity distribution - array shape (npix, npix)
    '''
    if centre is None:
        centre = [0, 0]
    # array to store permitivity in different square
    # assumed that background permitivity is 1 (and therefore difference with uniform will be 0)
    perm_sq = np.ones((npix, npix), dtype='f4')
    # initialises an array to store the coordinates of centres of squares (pixels)
    C_sq = np.empty((npix, npix, 2), dtype='f4')
    # initialising the j vector to prepare for tiling
    j = np.arange(npix)
    # tiling j to itself npix times (makes array shape (npix, npix))
    j = np.tile(j, (npix, 1))
    # i and j are transposes of each other    
    i = j
    j = j.T
    # assigning values to C_sq 
    C_sq[i, j, :] = np.transpose([a / 2 * ((2 * i + 1) / npix - 1) + centre[0], a / 2 * ((2 * j + 1) / npix - 1) + centre[1]])
    # return an empty array if there are no anomalies generated
    if anomaly is None:
    	return perm_sq - 1
    # putting anomalies on map one by one
    for l in range(len(anomaly)):
        if anomaly[l]['name'] == 'ellipse':
        # check what squares have their centres inside the ellipse and set permittivity values
            x = anomaly[l]['x']
            y = anomaly[l]['y']
            a_ = anomaly[l]['a']
            b = anomaly[l]['b']
            angle = anomaly[l]['angle']
            # equation for a rotated ellipse in 2d cartesians
            indices = np.sum(np.power([(np.cos(angle)*(C_sq[:, :, 0] - x) - np.sin(angle) * (C_sq[:, : , 1] - y))/a_, 
                                       (np.sin(angle)*(C_sq[:, :, 0] - x) + np.cos(angle) * (C_sq[:, : , 1] - y))/b], 2),
                                       axis=0) < 1
            # setting relative permittivity values
            perm_sq[indices] = anomaly[l]['perm']

        # check what squares are crossed by the line element and set their permittivity values to zero
        elif anomaly[l]['name'] == 'line':
            x = anomaly[l]['x'] # of centre
            y = anomaly[l]['y']
            theta = anomaly[l]['angle_line']
            length = anomaly[l]['len']
            # coordinates of endpoints of the line
            p_start = np.array([x + (length * np.cos(theta))/2, y + (length * np.sin(theta))/2])
            p_end = np.array([x - (length * np.cos(theta))/2, y - (length * np.sin(theta))/2])
            # find min and max x and y for any coordinates, so we have lower left and upper right corners of rectangle, whose diagonal is our line
            x_min_max = np.sort([x + (length * np.cos(theta))/2, x - (length * np.cos(theta))/2])
            y_min_max = np.sort([y + (length * np.sin(theta))/2, y - (length * np.sin(theta))/2])
            # checking whether pixel is in that rectangle by setting a limit on x and y of its centre
            if abs(y_min_max[0] - y_min_max[1]) < a / (npix/4):
            # the loop increases the allowed distances from the line if line is very close to horizontal
                index_1 = (C_sq[:,:,0] > x_min_max[0]) * (C_sq[:,:,0] < x_min_max[1]) * (C_sq[:,:,1] > y_min_max[0] - a / (npix * np.sqrt(2))) * (C_sq[:,:,1] < y_min_max[1] + a / (npix * np.sqrt(2)))
            elif abs(x_min_max[0] - x_min_max[1]) < a / (npix/4):
            # the loop increases the allowed distances from the line if line is very close to vertical
                index_1 =  (C_sq[:,:,0] > x_min_max[0] - a / (npix/4)) * (C_sq[:,:,0] < x_min_max[1] + a / (npix/4)) * (C_sq[:,:,1] > y_min_max[0]) * (C_sq[:,:,1] < y_min_max[1])
            else:
                index_1 = (C_sq[:,:,0] > x_min_max[0]) * (C_sq[:,:,0] < x_min_max[1]) * (C_sq[:,:,1] > y_min_max[0]) * (C_sq[:,:,1] < y_min_max[1])
            
            # checking whether distance from the centre to the line is smaller than the diagonal of the square
            indices = (np.absolute(np.cross(p_end - p_start,
                                   np.array([p_start[0] - C_sq[:, :, 0], p_start[1] - C_sq[:, :, 1]]).T)
                       / la.norm(p_end - p_start)) 
                       < a / (npix * np.sqrt(2)))
            indices = np.transpose(indices)
            # combining the two conditions 1)square in rectangle with line as diagonal and 2) distance to line smaller than half diagonal of pixel
            indices = np.multiply(indices, index_1)
            # setting permittivity values (relative)
            perm_sq[indices] = anomaly[l]['perm']
        elif anomaly[l]['name'] == 'triangle':
        #extracting the coordinatec of each of the vertices
            A = anomaly[l]['A']
            B = anomaly[l]['B']
            C = anomaly[l]['C']
        #for each point check whether the sum of the areas of the triangles formed by it and all combinations of
        #two of the vertices of the triangle is approximately equal to the area of the triangle
            index_tri = triangle_area([A[0], B[0], C_sq[:, :, 0]], [A[1], B[1], C_sq[:, :, 1]]) + triangle_area([B[0], C[0], C_sq[:, :, 0]], [B[1], C[1], C_sq[:, :, 1]]) + triangle_area([C[0], A[0], C_sq[:, :, 0]], [C[1], A[1], C_sq[:, :, 1]]) - triangle_area([A[0], B[0], C[0]], [A[1], B[1], C[1]]) < 0.01
        #set permitivity of triangle equal to the permitivity of the anomaly
            perm_sq[index_tri] = anomaly[l]['perm'] 
        elif anomaly[l]['name'] == 'gaussian_mixture':
            mu = anomaly[l]['mean']
            sigma = anomaly[l]['covariance']
            weightGauss = anomaly[l]['perm']
            permGauss, _ = generateContinuousConductivity(2., None, 1, mu, sigma, npix, weightGauss)
            perm_sq += permGauss
    perm_sq[perm_sq < 0.005] = 0.005
    perm_sq -= 1.
    #check whether any anomalies are too close to each of the edges
    indexx1 = (np.absolute(C_sq[:, :, 0] + a/2) < 0.15)
    indexx2 = (np.absolute(C_sq[:, :, 0] - a/2) < 0.15)
    indexy1 = (np.absolute(C_sq[:, :, 1] + a/2) < 0.15)
    indexy2 = (np.absolute(C_sq[:, :, 1] - a/2) < 0.15)
    #combine all edge conditions
    index = (indexx1 + indexx2 + indexy1 + indexy2)
    #check for permitivities close to 0
    index_p = perm_sq < -0.9995
    index = np.multiply(index_p, index)
    index = index > 0
    #set such conditions equal to the background to ensure existing solution to forward problem
    perm_sq[index] = 0.
    '''
    #optional plot conductivity distribution as a check
    plt.imshow(perm_sq, cmap=plt.cm.viridis, origin='lower', extent=[-1, 1, -1, 1])
    plt.colorbar()
    #plt.show()
    '''
    return perm_sq

def generate_anoms(a, b):
    '''
    a function that generates an array of a random number of dictionaries for the different anomalies; 

    also randomises the type and characteristics of the different anomalies

    takes: 

    a - length of side on x axis - float
    b - length of side on y axis - float

    returns:

    anoms - array of dictionaries with information about anomalies - array shape (number of anomalies); type(anoms[0]): object

    '''
    # random number of anomalies (between 0 and 2) with a given probability
    #n_anom = int(np.round_(np.absolute(np.random.normal(1.3, 0.35))))
    
    n_anom = int(np.random.choice([0,1,2,3], p=[0.07,0.45,0.38, 0.1]))
    #print(n_anom)
    # stop the function if n_anom == 0
    if n_anom == 0:
        return None
    # initialises an empty array of objects(dictionaries) with length n_anom
    anoms = np.empty((n_anom,), dtype='O')
    # chooses types of anomalies, (ellipse ( == 0) with probability 0.5, line ( == 1 ) with probability 0.1 and triangle ( == 2) with probability 0.4)
    names = np.random.choice([0, 1, 2, 3], n_anom, p=[0.4, 0.1, 0.4, 0.1])
    #names = 3 * np.ones(n_anom)
    # randomises characteristics of each anomaly and assigns dictionary to the anoms array
    for i in range(n_anom):
        if names[i] == 0:
            # x, y from -a to a or -b to b
            x = a * rand.random() - a/2
            y = b * rand.random() - b/2
            # a, b from 0 to a and 0 to b
            a_ = a * rand.random()
            b_ = b * rand.random()
            # angle from 0 to PI
            ang = np.pi * rand.random()
            #modulus to ensure positive values 0.25 times a random number + number from [0,1] based of power law of 1.2 + 0.5 base offset to prevent conductivities too low
            #main goal is to have a bias to higher conductivities since this is what we expect in real data
            perm = np.absolute(0.25 * np.random.randn() + np.random.power(1.2) + 0.5)
            # add anomaly to array
            anoms[i] = {'name': 'ellipse', 'x': x, 'y': y, 'a': a_, 'b': b_ ,'angle': ang, 'perm': perm}
        if names[i] == 1:
            # x, y from -a to a or -b to b
            x = a * rand.random() - a/2
            y = b * rand.random() - b/2
            # diagonal of the given line
            l = 0.5 * np.sqrt(a ** 2 + b ** 2)
            # total length of line anomaly ( with pdf proportional to 1.4) + constant length value
            #prevents existence of lines too short and overfitting of CNN
            length = l * (np.random.power(1.4) + 0.14)
            # angle from 0 to PI
            ang = np.pi * rand.random()
            # add anomaly to array
            anoms[i] = {'name': 'line', 'len': length, 'x': x, 'y': y, 'angle_line': ang, 'perm': 0.00001}
        if names[i] == 2:
            #two individual choices of triangles (one with random angles that are not too small with probability 70% and
            #one with angles 120/60 deg to capture the symmetry of the graphene sample)
            #tri_type = rand.choice([0, 1], p = [0.7, 0.3])
            tri_type = rand.choice([0,0,0,0,0,0,0,1,1,1])
            if tri_type == 0:
                #initialise cosines to be one rad.
                cosa = np.ones(3)
                while (np.absolute(cosa) - 0.5).all() > 0.01 and (np.absolute(cosa) - np.sqrt(3)/2).all() > 0.01:
                    #initialise coordinates of two of the points
                    Ax = a * rand.random() - a/2
                    Ay = b * rand.random() - b/2
                    Bx = a * rand.random() - a/2
                    By = b * rand.random() - b/2
                    #randomise which root to take since two such triangles will be possible for a set of 2 points
                    sign = rand.choice([-1, 1])
                    #calculate length of line connecting A/B
                    l = np.sqrt(np.power(Ax - Bx, 2) + np.power(Ay - By, 2))
                    #randomise length of second side of triangle (constant factor added to ensure existence of this specific triangle)
                    l2 = np.sqrt(3) * l / 2 + rand.random()
                    #calculate differences of y's of A/B
                    diff = np.absolute(Ay - By)
                    #initialise angle of 60 deg
                    ang = 1/3 * np.pi
                    #calculate angle needed to determine y coord of C 
                    beta = np.pi - np.arcsin(diff/l) - ang
                    #Calculate coordinates of C[x, y]
                    Cy = Ay - np.sin(beta)*l2
                    Cx = Ax - np.sqrt(np.power(l2, 2) - np.power(Ay - Cy, 2))
                    #iterate over created triangles to ensure desired result given lack of general formulation
                    a_sq = np.power(Ax - Bx, 2) + np.power(Ay - By, 2)
                    b_sq = np.power(Bx - Cx, 2) + np.power(By - Cy, 2)
                    c_sq = np.power(Cx - Ax, 2) + np.power(Cy - Ay, 2)
                    N1 = a_sq + b_sq - c_sq
                    D1 = 2 * np.sqrt(a_sq) * np.sqrt(b_sq)
                    N2 = b_sq + c_sq - a_sq
                    D2 = 2 * np.sqrt(b_sq) * np.sqrt(c_sq)
                    N3 = c_sq + a_sq - b_sq
                    D3 = 2 * np.sqrt(c_sq) * np.sqrt(a_sq)
                    cosa = ([np.true_divide(N1, D1), np.true_divide(N2, D2), np.true_divide(N3, D3)])

            elif tri_type == 1:
                #initialise cosine in order to enter loop
                cosa = 1
                #interate over created triangles to ensure no silver triangles
                while np.absolute(cosa) > np.sqrt(2)/2:
                    #initialise coordinates randomly
                    Ax = a * rand.random() - a/2
                    Ay = b * rand.random() - b/2
                    Bx = a * rand.random() - a/2
                    By = b * rand.random() - b/2
                    Cx = a * rand.random() - a/2
                    Cy = b * rand.random() - b/2
                    #calculate length of each side
                    a_sq = np.power(Ax - Bx, 2) + np.power(Ay - By, 2)
                    b_sq = np.power(Bx - Cx, 2) + np.power(By - Cy, 2)
                    c_sq = np.power(Cx - Ax, 2) + np.power(Cy - Ay, 2)
                    #calculate the maximum angle in triangle using cosine rule
                    N1 = a_sq + b_sq - c_sq
                    D1 = 2 * np.sqrt(a_sq) * np.sqrt(b_sq)
                    N2 = b_sq + c_sq - a_sq
                    D2 = 2 * np.sqrt(b_sq) * np.sqrt(c_sq)
                    N3 = c_sq + a_sq - b_sq
                    D3 = 2 * np.sqrt(c_sq) * np.sqrt(a_sq)
                    cosa = np.amax([np.true_divide(N1, D1), np.true_divide(N2, D2), np.true_divide(N3, D3)])
            #initialise perimitivity of triangle (constant value + pdf proportional to power 1.2 in the range [0,1] and another random constant factor)
            perm = np.absolute(0.25 * np.random.randn() + np.random.power(1.2) + 0.5)
            #return anomaly to the array of anomalies
            anoms[i] = {'name': 'triangle', 'A': [Ax, Ay], 'B': [Bx, By], 'C': [Cx, Cy], 'perm': perm}
        if names[i] == 3:
            mu, sigma = randomiseGaussianParam(a=2., centre=None, npix=64)
            weightGauss = rand.uniform(-.25, .25)
            #print("weightGauss =", weightGauss)
            anoms[i] = {'name': 'gaussian_mixture', 'mean': mu, 'covariance': sigma, 'perm': weightGauss}

    #return the whole dictionary
    return anoms
