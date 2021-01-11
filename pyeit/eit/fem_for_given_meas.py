# coding: utf-8
# pylint: disable=invalid-name, no-member, too-many-locals
# pylint: disable=too-many-instance-attributes
""" 2D/3D FEM routines """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
#from __future__ import division, absolute_import, print_function

from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
from cupyx.scipy import sparse as sp
from scipy.optimize import minimize
from .utils import eit_scan_lines
from time import time


cp.cuda.Device(0).use()
class Forward(object):
    """ FEM forward computing code """
    #@profile
    def __init__(self, mesh, el_pos, ne, z=None):
        """
        A good FEM forward solver should only depend on
        mesh structure and the position of electrodes
        NOTE: the nodes are assumed continuous numbered.

        Parameters
        ----------
        mesh : dict
            mesh structure
        el_pos : NDArray
            numbering of electrodes positions
        """
        # save all arrays on GPU using CuPy
        self.pts = cp.array(mesh['node'])
        self.tri = cp.array(mesh['element'])
        self.tri_perm = cp.array(mesh['perm'])
        self.el_pos = cp.array(el_pos)

        # reference electrodes [ref node should not be on electrodes]
        ref_el = 0
        while ref_el in self.el_pos:
            ref_el = ref_el + 1
        self.ref = ref_el

        # infer dimensions from mesh
        self.n_pts, self.n_dim = self.pts.shape
        self.n_tri, self.n_vertices = self.tri.shape
        self.ne = ne
        self.n_per_el = int(self.el_pos.size / self.ne)
        self.twoFromElectrode, self.nodeisElectrode, self.isValid = self.findTrianglesOnElectrodes()
        if z is None:
            self.z = 250. * cp.ones(self.ne)
        else:
            self.z = z
    #@profile
    def solve_eit(self, volt_mat=None, new_ind=None, ex_mat=None, step=1, perm=None, parser=None):
        """
        EIT simulation, generate perturbation matrix and forward v

        Parameters
        ----------
        ex_mat : NDArray
            numLines x n_el array, stimulation matrix
        step : int
            the configuration of measurement electrodes (default: adjacent)
        perm : NDArray
            Mx1 array, initial x0. must be the same size with self.tri_perm
        parser : str
            if parser is 'fmmu', within each stimulation pattern, diff_pairs
            or boundary measurements are re-indexed and started
            from the positive stimulus electrode
            if parser is 'std', subtract_row start from the 1st electrode

        Returns
        -------
        jac : NDArray
            number of measures x n_E complex array, the Jacobian
        v : NDArray
            number of measures x 1 array, simulated boundary measures
        b_matrix : NDArray
            back-projection mappings (smear matrix)
        """
        # deduce number of electrodes from number of electrodes and array of positions
        
        if ex_mat is None:
            ex_mat = cp.array(eit_scan_lines(self.ne, int(self.ne/2)))
        else:
            ex_mat = cp.array(ex_mat)
        # initialize/extract the step (takes both integer for constant and array for variable steps)
        if type(step) is int:
            step_arr = step * cp.ones(ex_mat.shape[0])
        elif type(step) is np.ndarray:
            if np.shape(step)[0] >= ex_mat.shape[0]:
                step_arr = cp.array(step)
            else:
                raise ValueError('Array is not long enough!')
        elif (volt_mat is not None) and (new_ind is not None):
            pass
        else:
            raise TypeError('Type of step is not int or ndarray!')
        # initialize permitivity
        if perm is None:
            perm0 = self.tri_perm
        elif cp.isscalar(perm):
            perm0 = cp.ones(self.n_tri, dtype=float)
        else:
            assert perm.shape == (self.n_tri,)
            perm0 = cp.array(perm)
        # if volt_mat is None or new_ind is None:
        #     volt_mat, new_ind = self.voltMeter(ex_mat, step_arr)
        #     print("activated 1")
        # elif len(volt_mat) == len(new_ind):
        #     ex_mat = disposeRepeated(ex_mat)
        #     new_ind = relabelNewInd(new_ind)
        #     #new_ind = cp.arange(ex_mat.shape[0])
        # else:
        #     raise ValueError('volt_mat and new_ind must be arrays (or lists/tuples) shape (N, 2) and (N) respectively. N is number of measurements.')
        ke = self.calculate_ke()
        #t_2 = pt()
        # calculate global stiffness matrix
        Ag = self.assemble_sparse(ke, self.tri, perm0, self.n_pts, ref=self.ref)
        #t_1 = pt()
        # calculate its inverse
        r_matrix = cp.linalg.inv(Ag)
        # extract values for electrodes
        r_el = r_matrix[self.n_pts:]
        #r_el = r_el.reshape(self.ne, self.n_per_el, r_el.shape[1])
        #r_el = cp.mean(r_el, axis=1)
        #t0 = pt()
        '''
        b = np.zeros((self.n_pts, ex_mat.shape[0]))
        r_el_np = cp.asnumpy(r_el)
        ex_mat_np = cp.asnumpy(ex_mat)
        i = 0
        for ex_line in ex_mat_np:
            b_el = self.optimise_currents(r_el_np, ex_line)
            b[ex_mat_np[i, 0] * self.n_per_el : (ex_mat_np[i, 0] + 1) * self.n_per_el, i] = b_el[:self.n_per_el]
            b[ex_mat_np[i, 1] * self.n_per_el : (ex_mat_np[i, 1] + 1) * self.n_per_el, i] = b_el[self.n_per_el:]
            i += 1
        b = cp.array(b)'''
        # set boundary conditions
        b = self._natural_boundary(ex_mat)
        # calculate f matrix (potential at nodes)
        f = cp.einsum('ij, jh', r_matrix, b)
        #t1 = pt()
        # calculate Jacobian
        jac_i = self.findJac(ex_mat, perm0, ke, f, r_el)
        #t2 = pt()
        f_el = f[:, self.n_pts:]
        #print(f_el.shape)
        #print(f_el[5])
        #f_el = f_el.reshape(f_el.shape[0], self.ne, self.n_per_el)
        #f_el = np.mean(f_el, axis=2)
        # generate all voltage measurements with given step
        if volt_mat is None or new_ind is None:
            volt_mat, new_ind = self.voltMeter(ex_mat, step_arr)
            #print("activated 2 if")
        elif len(volt_mat) == len(new_ind):
            volt_mat = cp.array(volt_mat, dtype='i1')
            new_ind = cp.array(new_ind, dtype='i1')
            #print("activated 2 elif")
        else:
            raise ValueError('volt_mat and new_ind must be arrays (or lists/tuples) shape (N, 2) and (N) respectively. N is number of measurements.')
        #t3 = pt()
        # find differences in voltage and in Jacobian at the measuring electrodes, since absolute values are not needed for conductivity reconstruction
        V = self.substractRow(f_el, volt_mat, new_ind)
        J = self.substractRow(jac_i, volt_mat, new_ind)
        #t4 = pt()
        # find smearing matrix from f (needed for backprojection)
        B = self.smearing(f, f_el, volt_mat, new_ind)
        #t5 = pt()
        # optional: check time performance
        '''
        print('kg takes:', t_1-t_2)
        print('inv takes:', t0-t_1)
        print('dot product takes:', t1-t0)
        print('Solve takes:', t2-t1)
        print('voltmeter takes:', t3-t2)
        print('subtract_row takes:', t4-t3)
        print('Smearing takes:', t5-t4)
        '''
        #print("New FEM voltages:\n", f)
        # return result as a tuple
        #print("\nex_mat:\n",ex_mat)
        ex_mat = ex_mat[new_ind]
        #print("\nex_mat:\n",ex_mat)
        #print("volt_mat:\n",volt_mat)
        #print("size ex_mat:\n",np.shape(ex_mat))
        #print("size volt_mat:\n",np.shape(volt_mat))
        meas = cp.concatenate((ex_mat, volt_mat), axis=1)
        #print("Ex mat and Volt mat:\n", meas)
        pde_result = namedtuple("pde_result", ['jac', 'v', 'b_matrix'])
        p = pde_result(jac=cp.asnumpy(J),
                       v=cp.asnumpy(V),
                       b_matrix=cp.asnumpy(B))
        #print(J.shape)
        return p, cp.asnumpy(meas), cp.asnumpy(new_ind)
        

    def _natural_boundary(self, ex_mat):
        """
        Notes
        -----
        Generate the Neumann boundary condition. In utils.py,
        you should note that ex_line is local indexed from 0...15,
        which need to be converted to global node number using el_pos.
        """
        drv_a_global_arr = ex_mat[:, 0].astype(int)
        drv_b_global_arr = ex_mat[:, 1].astype(int)

        row = cp.arange(ex_mat.shape[0])

        b = cp.zeros((self.ne, ex_mat.shape[0]))
        b[drv_a_global_arr, row] = 1e4
        b[drv_b_global_arr, row] = -1e4

        b_final = cp.zeros(( self.n_pts + self.ne, ex_mat.shape[0]))
        b_final[self.n_pts:, :] = b[:]
        return b_final

    def findJac(self, ex_mat, perm0, ke, f, r_el):
        '''
        Calculates Jacobian for all measurements

        takes:

        ex_mat - array shape (n_source/sinks, 2) - excitation matrix with source and sink for each measurement
        perm0 - array shape (n_triangles) - initial permittivity on each triangle
        ke - array shape (n_triangles, n_vertices, n_vertices) - stiffness on each element matrix
        f - array shape (n_nodes) - voltage on each node of mesh
        r_el - inverse of global stiffness matrix on electrodes

        returns:

        jac - array shape ( n_measurements, n_electrodes,n_triangles) - Jacobian for all measurements
        
        '''
        # initialise array for Jacobian
        jac = cp.zeros((ex_mat.shape[0], self.ne, self.n_tri), dtype=perm0.dtype)
        # calculating jacobian
        jac[:] = cp.einsum('ijk, jkp, ljp->lij', r_el[:, self.tri], ke, f[:, self.tri], optimize='optimal')
        #jac = cp.zeros((ex_mat.shape[0], self.ne, self.n_tri), dtype=perm0.dtype)
        #jac_all_el_pts = jac_all_el_pts.reshape((ex_mat.shape[0], self.ne, self.n_per_el, self.n_tri))
        #jac[:] = (1. / self.n_per_el) * np.sum(jac_all_el_pts, axis=2)
        return jac

    def substractRow(self, f_el, volt_mat, new_ind):
        '''
        Finds values of f_el for all pairs of measuring electrodes and finds the difference of f_el between its value at the two electrodes.
        
        takes:

        f_el - 1d array
        volt_mat - array shape (n_measurements, 2) - gives all volt. measurements
        new_ind - array shape (n_measurements) - helps with finding the relevant source-sink pair for each volt. measurement

        returns:

        v_diff - array shape (n_measurements) - difference in voltages or whatever f_el is

        '''
        # get first and second measuring electrode
        i = volt_mat[:, 0].astype(int)
        j = volt_mat[:, 1].astype(int)
        # perform subtraction
        v_diff = f_el[new_ind, i] - f_el[new_ind, j]

        return v_diff

    def smearing(self, f, f_el, volt_mat, new_ind):
        '''

        Produces B matrix by comparing voltages

        takes:

        f - array shape (n_nodes)
        f_el - array shape (n_electrodes)
        volt_mat - array shape (n_measurements, 2)
        new_ind - array shape (n_measurements)

        returns:

        b-matrix - array shape (n_measurements, n_nodes)

        '''
        i = cp.arange(len(volt_mat))
        f_volt0 = f_el[new_ind, volt_mat[:, 0].astype(int)]
        f_volt1 = f_el[new_ind, volt_mat[:, 1].astype(int)]
        min_fel = cp.minimum(f_volt0, f_volt1)
        max_fel = cp.maximum(f_volt0, f_volt1)
        b_matrix = cp.empty((len(volt_mat), self.n_pts+self.ne))
        b_matrix[:] = (min_fel[:, None] < f[new_ind]) & (f[new_ind] <= max_fel[:, None])

        return b_matrix
    #@profile
    def voltMeter(self, ex_mat, step_arr=None, parser=None):
        '''
        
        Returns all measurements with this step_arr and ex_mat

        takes:

        ex_mat - array shape (n_source/sinks, 2) - excitation matrix with source and sink for each measurement
        step_arr - array shape (n_source/sinks) - step between measuring electrodes for each source/sink pair
        parser - string

        returns:

        pair_mat - array shape (n_measurements, 2) - matrix with all possible meas. electrode combinations
        ind_new - array shape (n_measurements) - helper array

        '''        
        if step_arr is None:
            step_arr = 1 + cp.arange((ex_mat.shape[0])) % (self.ne)

        drv_a = ex_mat[:, 0]
        drv_b = ex_mat[:, 1]
        i0 = drv_a if parser == 'fmmu' else 0
        A = cp.arange(i0, i0 + self.ne)
        
        #M = cp.dot(cp.ones(ex_mat.shape[0])[:,None], A[None, :]) % self.ne
        #N = (M + step_arr[:, None]) % self.ne

        M = cp.arange(ex_mat.shape[0] * self.ne) % self.ne
        N = (M.reshape((ex_mat.shape[0], self.ne)) + step_arr[:, None]) % self.ne
        pair_mat = cp.stack((N.ravel(), M), axis=-1)

        #ind_new = cp.arange(pair_mat.shape[0]) % ex_mat.shape[0]
        ind_new = cp.arange(ex_mat.shape[0])        
        ind_new = cp.tile(ind_new, (self.ne, 1)).T.ravel()
        #print('before indtest', ind_new[20:70])
        nz2 = cp.where(pair_mat == drv_a[ind_new, None])
        nz3 = cp.where(pair_mat == drv_b[ind_new, None])
        #print(ind_new)
        ind_ = cp.arange(pair_mat.shape[0])
        ind_fin = cp.sum(ind_[:, None] == nz2[0][None], axis=1)
        ind_fin2 = cp.sum(ind_[:, None] == nz3[0][None], axis=1)

        ind_test = cp.less((ind_fin + ind_fin2), 0.5 * cp.ones(len(ind_fin)))

        pair_mat = pair_mat[ind_test, :]
        ind_new = ind_new[ind_test]
        #sort_index = cp.argsort(ind_new)

        #print('after indtest', ind_new[20:70])
        #meas = cp.concatenate((ex_mat[ind_new], pair_mat), axis=1)
        #print(meas[20:70])
        return pair_mat, ind_new

    #@profile
    def assemble_sparse(self, ke, tri, perm, n_pts, ref=0):
        '''
        function that assembles the global stiffness matrix from all element stiffness matrices

        takes:

        ke - stiffness on each element matrix - array shape (n_triangles, n_vertices, n_vertices)
        tri - array with all indices (in pts array) of triangle vertices - shape (num_triangles, 3)
        perm - array with permittivity in each element - array shape (num_triangles,)
        n_pts - number of nodes - int
        ref - electrode on which reference value is placed

        returns: 

        K - global stiffness matrix - (n_pts, n_pts)
        '''
        n_tri, n_vertices = tri.shape
        row = cp.tile(tri, (1, n_vertices))
        i = cp.array([0, 3, 6, 1, 4, 7, 2, 5, 8])
        row = row[:, i].ravel()
        col = cp.tile(tri, (n_vertices)).reshape((tri.shape[0] * tri.shape[1] * n_vertices))
        admittanceMatrixC2 = self.admittanceMatrixC2()
        data = cp.multiply(ke[:], perm[:, None, None])
        indexElectrode = cp.sort(self.tri[self.twoFromElectrode][self.isValid], axis=1)[:, 0] // self.n_per_el
        data[self.twoFromElectrode][self.isValid] = (data[self.twoFromElectrode][self.isValid] + ((1/self.z[indexElectrode]))[:, None, None] * admittanceMatrixC2)
        data = data.ravel()
        ind = cp.argsort(row)
        row = row[ind]
        col = col[ind]
        data = data[ind]
        unique, counts = cp.unique(row, return_counts=True)
        index_pointer = cp.zeros(n_pts + 1)
        sum_count = cp.cumsum(counts)
        index_pointer[unique[:]+1] = sum_count[:] 
        
        K = sp.csr_matrix((data, col, index_pointer), shape=(n_pts, n_pts), dtype=perm.dtype)

        K = K.toarray()

        A = cp.empty((self.n_pts + self.ne, self.n_pts + self.ne), dtype='f8')


        if 0 <= self.ref < n_pts:
            K[self.ref, :] = 0.
            K[:, self.ref] = 0.
            K[self.ref, self.ref] = 1.

        A[:self.n_pts, :self.n_pts] = K[:]
        admittanceMatrixE = self.admittanceMatrixE()
        A[self.n_pts:, :self.n_pts] = admittanceMatrixE.T
        A[:self.n_pts, self.n_pts:] = admittanceMatrixE
        A[self.n_pts:, self.n_pts:] = self.admittanceMatrixD()
        return A


    def calculate_ke(self):
        '''
        function that calculates the element stiffness matrix on each element

        takes:

        pts - array that contains the coordinates of all nodes in the mesh - shape (n_nodes, 2)
        tri - array with all indices (in pts array) of triangle vertices - shape (num_triangles, 3)
        
        returns:

        ke_array - an array of stiffness matrices for all elements (n_triangles, 3, 3)
        '''
        
        n_tri, n_vertices = self.tri.shape
        ke_array = cp.zeros((n_tri, n_vertices, n_vertices))
        coord = self.pts[self.tri[:,:]]
        ke_array[:] = self.triangle_ke(coord)
        return ke_array

    def triangle_ke(self, coord):
        '''
        function that calculates ke
        
        takes:
        
        coord - coordinates of each triangle's nodes - shape (n_triangles, 3, 2)
        
        returns:

        ke_array - an array of stiffness matrices for all elements (n_triangles, 3, 3)
        '''
        s = cp.array(coord[:, [2, 0, 1]] - coord[:, [1, 2, 0]]) # shape (n_tri, 3, 2)
        ke_matrix = cp.empty((len(coord), 3, 3))
        area = cp.abs(0.5 * self.det2x2(s[:, 0], s[:, 1]))
        ke_matrix[:] = cp.einsum('ijk,kli->ijl', s, s.T) / (4. * area[:, None, None])
        return ke_matrix

    def det2x2(self, s1, s2):
        """Calculate the determinant of a 2x2 matrix"""
        return s1[:, 0]*s2[:, 1] - s1[:, 1]*s2[:, 0]

    def shapeFunctionParameters(self):
        '''
        return arrays of parameters for all shape functions in all triangles on electrodes - shape ((n_el * n_per_el - 1), 3, 3)
        '''
        twoFromElectrode,_, isValid = self.findTrianglesOnElectrodes()
        #print(self.tri[twoFromElectrode][isValid])
        pointsTri = self.pts[self.tri[twoFromElectrode][isValid]] # shape ((n_el * n_per_el - 1), 3, 2)
        params = cp.empty((pointsTri.shape[0], 3, 3))
        params[:, :, 0] = cp.multiply(pointsTri[:, [1, 2, 0], 0], pointsTri[:, [2, 0, 1], 1]) - cp.multiply(pointsTri[:, [2, 0, 1], 0], pointsTri[:, [1, 2, 0], 1])
        params[:, :, 1] = pointsTri[:, [1, 2, 0], 1] - pointsTri[:, [2, 0, 1], 1]
        params[:, :, 2] = - (pointsTri[:, [1, 2, 0], 0] - pointsTri[:, [2, 0, 1], 0])
        
        return params

    def findTrianglesOnElectrodes(self):
        twoFromElectrode = (cp.sum(self.tri < self.ne * self.n_per_el, axis = 1) == 2)
        nodeisElectrode = self.tri[twoFromElectrode][self.tri[twoFromElectrode] < self.ne * self.n_per_el].reshape(self.tri[twoFromElectrode].shape[0], 2)
        isValid = ( nodeisElectrode[:, 0]//self.n_per_el - nodeisElectrode[:, 1]//self.n_per_el ) == 0
        return twoFromElectrode,nodeisElectrode, isValid
    
    def admittanceMatrixC2(self):
        '''
        compute matrix to calculate integral of two shape functions
        over the length of the electrode (assuming they are non-zero) - shape ((n_el * n_per_el - 1), 3, 3, 3)
        '''
        shapeParams = self.shapeFunctionParameters()
        whereIsZero = (cp.absolute(shapeParams) - 1e-12 < 0)
        indexZero = cp.where(whereIsZero)
        isConst = indexZero[2] == 2 # 1 for const x, 0 for const y
        zeroShapeFunc = cp.array(indexZero[1])
        indicesShapeFunctions = cp.outer(cp.ones(shapeParams.shape[0]), cp.arange(3))
        indicesShapeFunctions[:, ~zeroShapeFunc] = 0
        #print(indexZero)
        outerOfShapeFunc = cp.einsum('ijk, ipq -> ijpkq', shapeParams, shapeParams)
        #print(outerOfShapeFunc[0,0,0])
        integratingMatrix = cp.empty((outerOfShapeFunc.shape[0], outerOfShapeFunc.shape[3], outerOfShapeFunc.shape[4]))
        
        '''
        for i in range(20):
            #print(self.pts[nodeisElectrode[isValid], :][i])
            print(nodeisElectrode[isValid][i])
            print(self.tri[twoFromElectrode][isValid][i])
        #print(nodeisElectrode[isValid])'''
        sortedElNodeIndices = cp.sort(self.nodeisElectrode[self.isValid], axis=1)
        #print(sortedElNodeIndices)
        firstOrderY = cp.empty((outerOfShapeFunc.shape[0]))
        secondOrderY = cp.empty((outerOfShapeFunc.shape[0]))
        thirdOrderY = cp.empty((outerOfShapeFunc.shape[0]))
        constX = cp.ones((outerOfShapeFunc.shape[0], 3))
        firstOrderY[:] = self.pts[sortedElNodeIndices, :][:, 1, 1] - self.pts[sortedElNodeIndices, :][:, 0, 1] # y2 - y1
        secondOrderY[:] = 0.5 * (cp.power(self.pts[sortedElNodeIndices, :][:, 1, 1], 2) - cp.power(self.pts[sortedElNodeIndices, :][:, 0, 1], 2)) # 1/2 (y2^2 - y1^2)
        thirdOrderY[:] = 1./3. * (cp.power(self.pts[sortedElNodeIndices, :][:, 1, 1], 3) - cp.power(self.pts[sortedElNodeIndices, :][:, 0, 1], 3)) # 1/3 (y2^3 - y1^3)
        constX[:, 1] = self.pts[sortedElNodeIndices, :][:, 1, 0]
        constX = cp.einsum('ij, ik -> ijk', constX, constX)
        integratingMatrix[:, 0, 0] = firstOrderY[:]
        integratingMatrix[:, 0, 1] = firstOrderY[:]
        integratingMatrix[:, 1, 0] = firstOrderY[:]
        integratingMatrix[:, 0, 2] = secondOrderY[:]
        integratingMatrix[:, 2, 0] = secondOrderY[:]
        integratingMatrix[:, 1, 1] = firstOrderY[:]
        integratingMatrix[:, 1, 2] = secondOrderY[:]
        integratingMatrix[:, 2, 1] = secondOrderY[:]
        integratingMatrix[:, 2, 2] = thirdOrderY[:]
        integratingMatrix[:] = integratingMatrix * isConst[:, None, None]
        #print(integratingMatrix)
        #intm = cp.array(integratingMatrix)
        #print(constX)
        firstOrderX = cp.empty((outerOfShapeFunc.shape[0]))
        secondOrderX = cp.empty((outerOfShapeFunc.shape[0]))
        thirdOrderX = cp.empty((outerOfShapeFunc.shape[0]))
        constY = cp.ones((outerOfShapeFunc.shape[0], 3))
        firstOrderX[:] = self.pts[sortedElNodeIndices, :][:, 1, 0] - self.pts[sortedElNodeIndices, :][:, 0, 0] # x2 - x1
        secondOrderX[:] = 0.5 * (cp.power(self.pts[sortedElNodeIndices, :][:, 1, 0], 2) - cp.power(self.pts[sortedElNodeIndices, :][:, 0, 0], 2)) # 1/2 (x2^2 - x1^2)
        thirdOrderX[:] = 1./3. * (cp.power(self.pts[sortedElNodeIndices, :][:, 1, 0], 3) - cp.power(self.pts[sortedElNodeIndices, :][:, 0, 0], 3)) # 1/3 (x2^3 - x1^3)
        constY[:, 2] = self.pts[sortedElNodeIndices, :][:, 1, 1]
        constY = cp.einsum('ij, ik -> ijk', constY, constY)
        #print(constY)
        indicesConstX = cp.where(isConst)[0]
        indicesConstY = cp.where(~isConst)[0]
        #print(indicesConstY)
        integratingMatrix[indicesConstY, 0, 0] = firstOrderX[indicesConstY]
        integratingMatrix[indicesConstY, 0, 1] = secondOrderX[indicesConstY]
        integratingMatrix[indicesConstY, 1, 0] = secondOrderX[indicesConstY]
        integratingMatrix[indicesConstY, 0, 2] = firstOrderX[indicesConstY]
        integratingMatrix[indicesConstY, 2, 0] = firstOrderX[indicesConstY]
        integratingMatrix[indicesConstY, 1, 1] = thirdOrderX[indicesConstY]
        integratingMatrix[indicesConstY, 1, 2] = secondOrderX[indicesConstY]
        integratingMatrix[indicesConstY, 2, 1] = secondOrderX[indicesConstY]
        integratingMatrix[indicesConstY, 2, 2] = firstOrderX[indicesConstY]
        '''
        for i in range(40):
            print(intm[i])
            print(integratingMatrix[i])
            '''
        integratingMatrix[indicesConstX] = cp.multiply(integratingMatrix[indicesConstX], constX[indicesConstX])
        integratingMatrix[indicesConstY] = cp.multiply(integratingMatrix[indicesConstY], constY[indicesConstY])

        admittanceMatrix = cp.einsum('ijklm, ilm -> ijk', outerOfShapeFunc, integratingMatrix)
        
        admittanceMatrix[:] = cp.absolute(admittanceMatrix)
        admittanceMatrix[admittanceMatrix < 1e-18] = 0
        #admittanceMatrix2 = cp.sum(cp.multiply(outerOfShapeFunc, integratingMatrix[:, None, None, :, :]), axis = [3,4])

        #print(admittanceMatrix[:50,:50])
        #number_of_equal = cp.sum(cp.equal(cp.round_(admittanceMatrix, 16), cp.round_(admittanceMatrix2, 16)))
        #print(number_of_equal)
        #print(number_of_equal == admittanceMatrix.shape[0] * admittanceMatrix.shape[1] * admittanceMatrix.shape[2])
        return admittanceMatrix

    def admittanceMatrixE(self):
        shapeParams = self.shapeFunctionParameters()
        whereIsZero = (cp.absolute(shapeParams) - 1e-12 < 0)
        indexZero = cp.where(whereIsZero)
        isConst = indexZero[2] == 2 # 1 for const x, 0 for const y
        indicesConstX = cp.where(isConst)[0]
        indicesConstY = cp.where(~isConst)[0]
        sortedElNodeIndices = cp.sort(self.nodeisElectrode[self.isValid], axis=1)
        admittanceMatrixE = cp.zeros((self.n_pts, self.ne))
        shapeMatrix = cp.zeros((shapeParams.shape[0], shapeParams.shape[1], 2))
        integratingMatrix = cp.zeros((shapeParams.shape[0], 2))
        shapeMatrix[indicesConstY, :, 0] = shapeParams[indicesConstY, :, 0] + shapeParams[indicesConstY, :, 2] * self.pts[sortedElNodeIndices, :][indicesConstY, 1, 1][:, None]
        shapeMatrix[indicesConstY, :, 1] = shapeParams[indicesConstY, :, 1]
        shapeMatrix[indicesConstX, :, 0] = shapeParams[indicesConstX, :, 0] + shapeParams[indicesConstX, :, 1] * self.pts[sortedElNodeIndices, :][indicesConstX, 1, 0][:, None]
        shapeMatrix[indicesConstX, :, 1] = shapeParams[indicesConstX, :, 2]
        integratingMatrix[indicesConstY, 0] = self.pts[sortedElNodeIndices, :][indicesConstY, 1, 0] - self.pts[sortedElNodeIndices, :][indicesConstY, 0, 0]
        integratingMatrix[indicesConstY, 1] = 0.5 * (cp.power(self.pts[sortedElNodeIndices, :][indicesConstY, 1, 0], 2) - cp.power(self.pts[sortedElNodeIndices, :][indicesConstY, 0, 0], 2))
        integratingMatrix[indicesConstX, 0] = self.pts[sortedElNodeIndices, :][indicesConstX, 1, 1] - self.pts[sortedElNodeIndices, :][indicesConstX, 0, 1]
        integratingMatrix[indicesConstX, 1] = 0.5 * (cp.power(self.pts[sortedElNodeIndices, :][indicesConstX, 1, 1], 2) - cp.power(self.pts[sortedElNodeIndices, :][indicesConstX, 0, 1], 2))
        #print(integratingMatrix.shape)
        integrals = cp.einsum('ijk, ik -> ij', shapeMatrix, integratingMatrix)
        integrals[:] = cp.absolute(integrals)
        #integr = cp.sum(cp.multiply(shapeMatrix, integratingMatrix[:, None]), axis=2)

        #print(cp.sum(cp.round_(integrals, 16) == cp.round_(integr, 16)))

        indexElectrode = sortedElNodeIndices[:, 0] // self.n_per_el
        #print(indexElectrode)
        integrals = - integrals / self.z[indexElectrode][:, None, None]
        integrals = integrals.ravel()
        indexElectrode = cp.tile(indexElectrode, (self.n_per_el, 1)).T.ravel()
        #print(self.tri[twoFromElectrode][isValid])
        indexNode = self.tri[self.twoFromElectrode][self.isValid].ravel()
        
        #admittanceMatrixE [self.tri[twoFromElectrode][isValid].ravel(), indexElectrode] += integrals.ravel()
        indSort = cp.argsort(indexNode)
        indexNode = indexNode[indSort]
        indexElectrode = indexElectrode[indSort]
        integrals = integrals[indSort]

        unique, counts = cp.unique(indexNode, return_counts=True)
        #print("number of unique entries", unique.shape)
        #print("counts \n", counts)
        index_pointer = cp.zeros(self.n_pts + 1)
        sum_count = cp.cumsum(counts)
        #print(sum_count)
        index_pointer[unique[:]+1] = sum_count[:]
        #print(index_pointer)
        nonzeroes = cp.nonzero(index_pointer)[0]
        #print(nonzeroes)
        mask = cp.zeros(index_pointer.shape[0], dtype='b1')
        mask[nonzeroes] = True
        mask[0] = True
        zeroes = cp.where(~mask)[0]
        #time_loop = time()
        while (index_pointer[1:]==0).any():
            index_pointer[zeroes] = index_pointer[zeroes - 1]
        '''for i in range(index_pointer.shape[0]):
            if i == 0:
                continue
            elif index_pointer[i] == 0:
                index_pointer[i] = index_pointer[i-1]'''
        #print('time for loop ',time()-time_loop)
        index_pointer2 = cp.arange(self.n_pts + 1)
        #print('indexEl', indexElectrode)
        #print(index_pointer.shape)
        admittanceMatrixE = sp.csr_matrix((integrals, indexElectrode, index_pointer), shape=(self.n_pts, self.ne), dtype=integrals.dtype)
        adm = admittanceMatrixE.toarray();
        #print(integrals)
        #print(indexNode)
        #print(indexElectrode)
        #a = (sortedElNodeIndices[0,0])
        #print(adm[4])
        # print(adm[:,1])
        #print('sum zeroes ',cp.sum(adm>0))
        return adm; 

    def admittanceMatrixD(self):
        all_el_nodes_coords = self.pts[:(self.ne * self.n_per_el)].reshape((self.ne, self.n_per_el, 2))
        lengths = cp.linalg.norm((all_el_nodes_coords[:, 0] - all_el_nodes_coords[:, (self.n_per_el - 1)]), axis=1)
        admittanceMatrixD = cp.diag(lengths/self.z)
        return admittanceMatrixD

def disposeRepeated(ex_mat):
    #get rid of all repeated source sink pairs in ex_mat (information about them will be kept in new_ind array)
    
    index_XM = cp.sum(cp.equal(ex_mat[:, None, :], ex_mat[None]), axis=2) == 2
    indices = cp.where(index_XM)
    ind = (indices[0] > indices[1])
    indices = [indices[0][ind], indices[1][ind]]
    i = cp.ones(len(ex_mat), dtype='i4')
    indices = cp.unique(indices[0])
    i[indices] = 0
    i= i.astype(bool)
    ex_mat = ex_mat[i]
    
    return ex_mat

def relabelNewInd(new_ind):
    #make new_ind consistent with new ex_mat indices
    ind = new_ind[:, None] == new_ind[None]
    new_ind = cp.argmax(ind, axis=0)
    repeated_ind = (new_ind != cp.arange(len(new_ind)))
    cumul = cp.cumsum(repeated_ind)
    cumul[1:] = cumul[:-1]
    cumul[0] = 0
    new_ind[~repeated_ind] -= cumul[~repeated_ind]
    new_ind[repeated_ind] -= cumul[new_ind[repeated_ind]]

    return new_ind
