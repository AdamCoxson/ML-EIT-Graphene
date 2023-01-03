# -*- coding: utf-8 -*-
# pylint: disable=invalid-name, no-member, too-many-arguments
""" wrapper function of distmesh for EIT """
# Copyright (c) Benyuan Liu. All rights reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function

import numpy as np

from .distmesh import build
from .mesh_circle import MeshCircle
from .utils import check_order
from .shape import unit_circle, unit_ball, area_uniform
from .shape import fix_points_fd, fix_points_ball

def multivariateGaussian(x, mu, sigma, normalised=False):
    if normalised:
        denominator = 1. / (2 * np.pi * np.sqrt(np.linalg.det(sigma)))
    else:
        denominator = 1.
    x_centred = x - mu
    #path = np.einsum_path('ij, jk, ki->i', x_centred, np.linalg.inv(sigma), x_centred.T, optimize='optimal')[0]
    numerator = np.exp(-0.5 * np.einsum('ij, jk, ki->i', x_centred, np.linalg.inv(sigma), x_centred.T, optimize='optimal'))
    return numerator / denominator

def triangle_area(x, y):
    return 0.5 * np.absolute(x[0] * (y[1] - y[2]) + x[1] * (y[2] - y[0]) + x[2] * (y[0] - y[1]))

def create(n_el=16, fd=None, fh=None, p_fix=None, bbox=None, h0=0.1):
    """
    wrapper for pyEIT interface

    Parameters
    ----------
    n_el : int, optional
        number of electrodes
    fd : function
        distance function
    fh : function
        mesh size quality control function
    p_fix : NDArray
        fixed points
    bbox : NDArray
        bounding box
    h0 : float, optional
        initial mesh size

    Returns
    -------
    dict
        {'element', 'node', 'perm'}
    """
    if bbox is None:
        bbox = [[-1, -1], [1, 1]]
    # infer dim
    bbox = np.array(bbox)
    n_dim = bbox.shape[1]
    if n_dim not in [2, 3]:
        raise TypeError('distmesh only supports 2D or 3D')
    if bbox.shape[0] != 2:
        raise TypeError('please specify lower and upper bound of bbox')

    if n_dim == 2:
        if fd is None:
            fd = unit_circle
        if p_fix is None:
            p_fix = fix_points_fd(fd, n_el=n_el)
    elif n_dim == 3:
        if fd is None:
            fd = unit_ball
        if p_fix is None:
            p_fix = fix_points_ball(n_el=n_el)

    if fh is None:
        fh = area_uniform

    # 1. build mesh
    p, t = build(fd, fh, pfix=p_fix, bbox=bbox, h0=h0)
    # 2. check whether t is counter-clock-wise, otherwise reshape it
    t = check_order(p, t)
    # 3. generate electrodes, the same as p_fix (top n_el)
    el_pos = np.arange(n_el)
    # 4. init uniform element permittivity (sigma)
    perm = np.ones(t.shape[0], dtype=np.float)
    # 5. build output structure
    mesh = {'element': t,
            'node': p,
            'perm': perm}
    return mesh, el_pos, p_fix


def set_perm(mesh, anomaly=None, background=None):
    """ wrapper for pyEIT interface

    Note
    ----
    update permittivity of mesh, if specified.

    Parameters
    ----------
    mesh : dict
        mesh structure
    anomaly : dict, optional
        anomaly is a dictionary (or arrays of dictionary) contains,
        {'x': val, 'y': val, 'd': val, 'perm': val}
        all permittivity on triangles whose distance to (x,y) are less than (d)
        will be replaced with a new value, 'perm' may be a complex value.
    background : float, optional
        set background permittivity

    Returns
    -------
    dict
        updated mesh structure
    """
    pts = mesh['element']
    tri = mesh['node']
    perm = mesh['perm'].copy()
    tri_centers = np.mean(tri[pts], axis=1)

    # this code is equivalent to:
    # >>> N = np.shape(tri)[0]
    # >>> for i in range(N):
    # >>>     tri_centers[i] = np.mean(pts[tri[i]], axis=0)
    # >>> plt.plot(tri_centers[:,0], tri_centers[:,1], 'kx')
    n = np.size(mesh['perm'])

    # reset background if needed
    if background is not None:
        perm = background * np.ones(n)

    # change dtype to 'complex' for complex-valued permittivity
    if anomaly is not None:
        for attr in anomaly:
            if np.iscomplex(attr['perm']):
                perm = perm.astype('complex')
                break

    # assign anomaly values (for elements in regions)
    if anomaly is not None:
        for _, attr in enumerate(anomaly):
            # find elements whose distance to (cx,cy) is smaller than d
            ''''if 'z' in attr:
                index = np.sqrt((tri_centers[:, 0] - attr['x'])**2 +
                                (tri_centers[:, 1] - attr['y'])**2 +
                                (tri_centers[:, 2] - attr['z'])**2) < d
            else:
                index = np.sqrt((tri_centers[:, 0] - attr['x'])**2 +
                                (tri_centers[:, 1] - attr['y'])**2) < d'''
            #index = (((tri_centers[:, 0] - attr['x'])/attr['a'])**2 +
            #                ((tri_centers[:, 1] - attr['y'])/attr['b'])**2) < 1
            # checking if anomaly is ellipse or line and assigning the values
            if attr['name'] == 'ellipse':
                x = attr['x']
                y = attr['y']
                a = attr['a']
                b = attr['b']
                angle = attr['angle']
                # checking if the centre of each triangle is in the ellipse (2d rotated ellipse equation in cartesians)
                index = np.sum(np.power([(np.cos(angle)*(tri_centers[:, 0] - x) - np.sin(angle) * (tri_centers[:, 1] - y))/a, 
                                        (np.sin(angle)*(tri_centers[:, 0] - x) + np.cos(angle) * (tri_centers[:, 1] - y))/b], 2),
                                        axis=0) < 1
                # set permittivity in triangles with centres inside ellipse
                perm[index] = attr['perm']
            elif attr['name'] == 'line':
                x = attr['x']
                y = attr['y']
                length = attr['len']
                theta = attr['angle_line']
                # get rid of poles by setting tan(theta) to 0.0001 instead of 0, so as not to divide by zero
                if theta != 0 and theta != 0.5*np.pi and theta != np.pi:
                    tan_th = np.tan(theta)
                elif theta == 0 or theta == np.pi:
                    tan_th = 0.0001
                elif theta == 0.5*np.pi:
                    tan_th = np.tan((0.5*np.pi)+0.0001)
                # finding intercept of the line equation (b = y - m*x), where m = tan(theta)
                intercept = y - tan_th*x
                # coordinates of endpoints of the line
                p_start = np.array([x + (length * np.cos(theta))/2, y + (length * np.sin(theta))/2])
                p_end = np.array([x - (length * np.cos(theta))/2, y - (length * np.sin(theta))/2])
                # find min and max x and y for any coordinates, so we have lower left and upper right corners of rectangle, whose diagonal is our line
                x_min_max = np.sort([x + (length * np.cos(theta))/2, x - (length * np.cos(theta))/2])
                y_min_max = np.sort([y + (length * np.sin(theta))/2, y - (length * np.sin(theta))/2])
                # checking whether triangle is in that rectangle by setting a limit on x and y of its centre
                if abs(y_min_max[0] - y_min_max[1]) < 2 / np.sqrt(float(len(tri_centers))):
                    # the loop increases the allowed distances from the line if line is very close to horizontal
                    index_sq = (tri_centers[:, 0] > x_min_max[0]) * (tri_centers[:,0] < x_min_max[1]) * (tri_centers[:,1] > y_min_max[0] - 2 / np.sqrt(float(len(tri_centers)))) * (tri_centers[:,1] < y_min_max[1] + 2 / np.sqrt(float(len(tri_centers))))
                elif abs(x_min_max[0] - x_min_max[1]) < 2 / np.sqrt(float(len(tri_centers))):
                    # the loop increases the allowed distances from the line if line is very close to vertical
                    index_sq =  (tri_centers[:,0] > x_min_max[0] - 2 / np.sqrt(float(len(tri_centers)))) * (tri_centers[:,0] < x_min_max[1] + 2 / np.sqrt(float(len(tri_centers)))) * (tri_centers[:,1] > y_min_max[0]) * (tri_centers[:,1] < y_min_max[1])
                else:
                    index_sq = (tri_centers[:,0] > x_min_max[0]) * (tri_centers[:,0] < x_min_max[1]) * (tri_centers[:,1] > y_min_max[0]) * (tri_centers[:,1] < y_min_max[1])
                # checking whether line passes through any triangles and setting their permitivity to the anomaly
                y_min_args=np.argmin(tri[pts[:, :], 1], axis=1)
                y_max_args=np.argmax(tri[pts[:, :], 1], axis=1)
                x_min_args=np.argmin(tri[pts[:, :], 0], axis=1)
                x_max_args=np.argmax(tri[pts[:, :], 0], axis=1)
                dy_max=np.empty((len(pts), 2))
                dy_min=np.empty((len(pts), 2))
                dx_max=np.empty((len(pts), 2))
                dy_min=np.empty((len(pts), 2))
                # creating a temporary array to help with indexing (not to use for loops)                
                temp_index = np.vstack((np.arange(len(pts)), y_max_args)).T
                # checking whether min and max x and y of each triangle are on the left or right of the line using matrices
                dy_max=tri[pts[temp_index[:, 0], temp_index[:, 1]], 1] - tan_th*tri[pts[temp_index[:, 0], temp_index[:, 1]], 0] - intercept
                temp_index = np.vstack((np.arange(len(pts)), y_min_args)).T
                dy_min=tri[pts[temp_index[:, 0], temp_index[:, 1]], 1] - tan_th*tri[pts[temp_index[:, 0], temp_index[:, 1]], 0] - intercept
                temp_index = np.vstack((np.arange(len(pts)), x_max_args)).T
                dx_max=tri[pts[temp_index[:, 0], temp_index[:, 1]], 0] - 1/tan_th*(tri[pts[temp_index[:, 0], temp_index[:, 1]], 1] - intercept)
                temp_index = np.vstack((np.arange(len(pts)), x_min_args)).T
                dx_min=tri[pts[temp_index[:, 0], temp_index[:, 1]], 0] - 1/tan_th*(tri[pts[temp_index[:, 0], temp_index[:, 1]], 1] - intercept)
                del temp_index
                # checkin if y_max is on top and y_min below the line to check if the line passes through triangle
                y_index = np.multiply(dy_max, dy_min) < 0
                # same for x_max and x_min
                x_index = np.multiply(dx_max, dx_min) < 0
                # if either of the y_max/y_min or x_max/x_min pairs lie on different sides of the line, the triangle is split, also has to be in the rectangle
                # in which our line is one of the diagonals                
                index = (y_index + x_index) * index_sq
                # setting the permittivity of elements which satisfy the conditions above to the anomalous permittivity                
                perm[index] = attr['perm']
            # also introduce a third anomaly - triangle
            elif attr['name'] == 'triangle':
                # inherit the three vertices from the dictionary
                A = attr['A']
                B = attr['B']
                C = attr['C']
                # check if the sum of the areas of the triangles formed between the vertices of the triangle and a point is bigger than triangle area (to check whether it's inside it)
                index_tri = triangle_area([A[0], B[0], tri_centers[:, 0]], [A[1], B[1], tri_centers[:, 1]]) + triangle_area([B[0], C[0], tri_centers[:, 0]], [B[1], C[1], tri_centers[:, 1]]) + triangle_area([C[0], A[0], tri_centers[:, 0]], [C[1], A[1], tri_centers[:, 1]]) - triangle_area([A[0], B[0], C[0]], [A[1], B[1], C[1]]) < 0.01
                # set permittivity of ones inside to the permittivity of the anomaly
                perm[index_tri] = attr['perm']
            elif attr['name'] == 'gaussian_mixture':
                mu = attr['mean'][0]
                sigma = attr['covariance'][0]
                amplitude = attr['perm']
                perm += (amplitude) * multivariateGaussian(tri_centers, mu, sigma)
        perm[perm < 0.005] = 0.005
        # setting the permittivity of elements within 0.15 of the border (the frame) to the background 
        indexx1 = (np.absolute(tri[pts[:, :], 0] + 1) < 0.15)
        indexx2 = (np.absolute(tri[pts[:, :], 0] - 1) < 0.15)
        indexy1 = (np.absolute(tri[pts[:, :], 1] + 1) < 0.15)
        indexy2 = (np.absolute(tri[pts[:, :], 1] - 1) < 0.15)
        # triangles only have to be next to one border for them to be in the frame of the sample (logical gate or is equivalent to summation)
        indices = (indexx1 + indexx2 + indexy1 + indexy2)
        # index to check whether permitivity is really low (<0.1)
        index_p = perm < 0.0005
        # we need only one of the vertices of the triangle to be in the frame for its permittivity to be set to the background
        indices = np.sum(indices, axis=1)
        # even though indices should be a Boolean array, numpy.sum took its sum as if it was an int array, therefore we get values bigger than zero
        indices = indices > 0
        # only if permittivity is really low set it to background
        indices *= index_p
        # set permittivity to background
        perm[indices] = 1.

    mesh_new = {'node': tri,
                'element': pts,
                'perm': perm}
    return mesh_new


def layer_circle(n_el=16, n_fan=8, n_layer=8):
    """ generate mesh on unit-circle """
    model = MeshCircle(n_fan=n_fan, n_layer=n_layer, n_el=n_el)
    p, e, el_pos = model.create()
    perm = np.ones(e.shape[0])

    mesh = {'element': e,
            'node': p,
            'perm': perm}
    return mesh, el_pos