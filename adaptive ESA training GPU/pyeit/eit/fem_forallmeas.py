# coding: utf-8
# pylint: disable=invalid-name, no-member, too-many-locals
# pylint: disable=too-many-instance-attributes
""" 2D/3D FEM routines """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
#from __future__ import division, absolute_import, print_function

from collections import namedtuple
import numpy as np
import cupy as cp
from cupyx.scipy import sparse as sp
from .utils import eit_scan_lines
from time import process_time as pt


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
        self.pts = np.array(mesh['node'])
        self.tri = np.array(mesh['element'])
        self.tri_perm = np.array(mesh['perm'])
        self.el_pos = np.array(el_pos)

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
            self.z = 250. * np.ones(self.ne)
        else:
            self.z = z
    #@profile
    def solve_eit(self, ex_mat=None, step=1, perm=None, parser=None):
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
        # initialize/extract the scan lines (default: apposition)


        if ex_mat is None:
            ex_mat = np.array(eit_scan_lines(20, 10))
        else:
            ex_mat = np.array(ex_mat)

        if type(step) is int:
            step_arr = step * np.ones(20)
        elif type(step) is np.ndarray:
            if np.shape(step)[0] >= ex_mat.shape[0]:
                step_arr = np.array(step)
            else:
                raise ValueError('Array is not long enough!')
        else:
            raise TypeError('Type of step is not int or ndarray!')
        # initialize the permittivity on element
        if perm is None:
            perm0 = self.tri_perm
        elif np.isscalar(perm):
            perm0 = np.ones(self.n_tri, dtype=float)
        else:
            assert perm.shape == (self.n_tri,)
            perm0 = np.array(perm)

        # calculate f and Jacobian iteratively over all stimulation lines
        #volt_mat = -1. * np.ones()
       
        ke = self.calculate_ke()
        #t_2 = pt()
        Ag = self.assemble_sparse(ke, self.tri, perm0, self.n_pts, ref=self.ref)
        #t_1 = pt()
        r_matrix = np.linalg.inv(Ag)
        #r_matrix = np.array(r_matrix)
        r_el = r_matrix[self.n_pts:]
        #t0 = pt()
        b = self._natural_boundary(ex_mat)
        f = np.dot(r_matrix, b).T
        #t1 = pt()
        jac_i = self.findJac(ex_mat, perm0, ke, f, r_el)
        #t2 = pt()
        f_el = f[:, self.n_pts:]
        volt_mat, ex_mat, new_ind = self.voltMeter(ex_mat, step_arr)
        #t3 = pt()
        V = self.substractRow(f_el, volt_mat, new_ind)
        J = self.substractRow(jac_i, volt_mat, new_ind)
        #t4 = pt()
        B = self.smearing(f, f_el, volt_mat, new_ind)
        #t5 = pt()
        '''
        print('kg takes:', t_1-t_2)
        print('inv takes:', t0-t_1)
        print('dot product takes:', t1-t0)
        print('Solve takes:', t2-t1)
        print('voltmeter takes:', t3-t2)
        print('subtract_row takes:', t4-t3)
        print('Smearing takes:', t5-t4)
        '''
        #print("\nex_mat:\n",ex_mat)
        #print("volt_mat:\n",volt_mat)
        #print("size ex_mat:\n",np.shape(ex_mat))
        #print("size volt_mat:\n",np.shape(volt_mat))
        meas = np.concatenate((ex_mat, volt_mat), axis=1)
        #print("Ex mat and Volt mat:\n", meas)
        pde_result = namedtuple("pde_result", ['jac', 'v', 'b_matrix'])
        p = pde_result(jac=J,
                       v=V,
                       b_matrix=B)
     
        return p, meas, new_ind


        
    def solveAnomaly(self, ex_mat=None, step=1, perm=None, parser=None):
        if ex_mat is None:
            ex_mat = np.array(eit_scan_lines(20, 8))
        else:
            ex_mat = np.array(ex_mat)

        if type(step) is int:
            step_arr = step * np.ones(20)
        elif type(step) is np.ndarray:
            if np.shape(step)[0] >= ex_mat.shape[0]:
                step_arr = np.array(step)
            else:
                raise ValueError('Array is not long enough!')
        else:
            raise TypeError('Type of step is not int or ndarray!')
        # initialize the permittivity on element
        if perm is None:
            perm0 = self.tri_perm
        elif np.isscalar(perm):
            perm0 = np.ones(self.n_tri, dtype=float)
        else:
            assert perm.shape == (self.n_tri,)
            perm0 = np.array(perm)

        # calculate f and Jacobian iteratively over all stimulation lines
        #volt_mat = -1. * np.ones()
       
        ke = self.calculate_ke()
        #t_2 = pt()
        Ag = self.assemble_sparse(ke, self.tri, perm0, self.n_pts, ref=self.ref)
        #t_1 = pt()
        r_matrix = np.linalg.inv(Ag)
        #r_matrix = np.array(r_matrix)
        r_el = r_matrix[self.n_pts:]
        #t0 = pt()
        b = self._natural_boundary(ex_mat)
        f = np.dot(r_matrix, b).T
        #t1 = pt()
        #t2 = pt()
        f_el = f[:, self.n_pts:]
        volt_mat, ex_mat, new_ind = self.voltMeter(ex_mat, step_arr)
        #t3 = pt()
        V = self.substractRow(f_el, volt_mat, new_ind)
        #t4=pt()
        '''
        print('kg takes:', t_1-t_2)
        print('inv takes:', t0-t_1)
        print('dot product takes:', t1-t0)
        print('Solve takes:', t2-t1)
        print('voltmeter takes:', t3-t2)
        print('subtract_row takes:', t4-t3)
        '''
        meas = np.concatenate((ex_mat, volt_mat), axis=1)
     
        return V, meas, new_ind

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
        jac = np.zeros((ex_mat.shape[0], self.ne, self.n_tri), dtype=perm0.dtype)
        # calculating jacobian
        jac[:] = np.einsum('ijk, jkp, ljp->lij', r_el[:, self.tri], ke, f[:, self.tri], optimize='optimal')
        #jac = np.zeros((ex_mat.shape[0], self.ne, self.n_tri), dtype=perm0.dtype)
        #jac_all_el_pts = jac_all_el_pts.reshape((ex_mat.shape[0], self.ne, self.n_per_el, self.n_tri))
        #jac[:] = (1. / self.n_per_el) * np.sum(jac_all_el_pts, axis=2)
        return jac

    def _natural_boundary(self, ex_mat):
        """
        Notes
        -----
        Generate the Neumann boundary condition. In utils.py,
        you should note that ex_line is local indexed from 0...15,
        which need to be converted to global node number using el_pos.
        """
        drv_a_global_arr = self.el_pos[ex_mat[:, 0]]
        drv_b_global_arr = self.el_pos[ex_mat[:, 1]]

        row = np.arange(ex_mat.shape[0])

        b = np.zeros((self.ne, ex_mat.shape[0]))
        b[drv_a_global_arr, row] = 1e4
        b[drv_b_global_arr, row] = -1e4

        b_final = np.zeros(( self.n_pts + self.ne, ex_mat.shape[0]))
        b_final[self.n_pts:, :] = b[:]
        return b_final

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
        i = np.arange(len(volt_mat))
        f_volt0 = f_el[new_ind, volt_mat[:, 0].astype(int)]
        f_volt1 = f_el[new_ind, volt_mat[:, 1].astype(int)]
        min_fel = np.minimum(f_volt0, f_volt1)
        max_fel = np.maximum(f_volt0, f_volt1)
        b_matrix = np.empty((len(volt_mat), self.n_pts+self.ne))
        b_matrix[:] = (min_fel[:, None] < f[new_ind]) & (f[new_ind] <= max_fel[:, None])

        return b_matrix
    #@profile
    def voltMeter(self, ex_mat, step_arr=None, parser=None):

        pair_mat = volt_mat(self.ne)
        print(len(pair_mat))
        print("pair_mat:\n", pair_mat)
        srcsinkpairs = ex_mat.shape[0]
        print("srcsinkpairs:\n", srcsinkpairs)

        pair_mat = np.tile(pair_mat, (int(ex_mat.shape[0]), 1, 1))
        pair_mat = pair_mat.reshape((pair_mat.shape[0] * pair_mat.shape[1], 2))
        print(len(pair_mat))
        ind_new = np.arange(pair_mat.shape[0]) // srcsinkpairs

        nz2 = np.where(pair_mat == ex_mat[:, 0][ind_new, None])
        nz3 = np.where(pair_mat == ex_mat[:, 1][ind_new, None])

        ind_ = np.arange(pair_mat.shape[0])
        ind_fin = np.sum(ind_[:, None] == nz2[0][None], axis=1)
        ind_fin2 = np.sum(ind_[:, None] == nz3[0][None], axis=1)

        ind_test = np.less((ind_fin + ind_fin2), 0.5 * np.ones(len(ind_fin)))

        pair_mat = pair_mat[ind_test, :]
        ind_new = ind_new[ind_test]

        return pair_mat, ex_mat[ind_new], ind_new


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
        row = np.tile(tri, (1, n_vertices))
        i = np.array([0, 3, 6, 1, 4, 7, 2, 5, 8])
        row = row[:, i].ravel()
        col = np.tile(tri, (n_vertices)).reshape((tri.shape[0] * tri.shape[1] * n_vertices))
        admittanceMatrixC2 = self.admittanceMatrixC2()
        data = np.multiply(ke[:], perm[:, None, None])
        indexElectrode = np.sort(self.tri[self.twoFromElectrode][self.isValid], axis=1)[:, 0] // self.n_per_el
        data[self.twoFromElectrode][self.isValid] = (data[self.twoFromElectrode][self.isValid] + ((1/self.z[indexElectrode]))[:, None, None] * admittanceMatrixC2)
        data = data.ravel()
        ind = np.argsort(row)
        row = row[ind]
        col = col[ind]
        data = data[ind]
        unique, counts = np.unique(row, return_counts=True)
        index_pointer = np.zeros(n_pts + 1)
        sum_count = np.cumsum(counts)
        index_pointer[unique[:]+1] = sum_count[:] 
        
        data = cp.array(data)
        col = cp.array(col)
        index_pointer = cp.array(index_pointer)
        #data =  [item for sublist in data for item in sublist]
        #col = col.ravel
        #col =  [item for sublist in col for item in sublist]
        #index_pointer = np.array(index_pointer).astype(np.int32)
        #index_pointer = index_pointer.ravel()
        #index_pointer = [item for sublist in index_pointer for item in sublist]
        K = sp.csr_matrix((data, col, index_pointer), shape=(n_pts, n_pts), dtype=perm.dtype)
        K = K.toarray()
        K = cp.asnumpy(K)
        A = np.empty((self.n_pts + self.ne, self.n_pts + self.ne), dtype='f8')
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
        ke_array = np.zeros((n_tri, n_vertices, n_vertices))
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
        s = np.array(coord[:, [2, 0, 1]] - coord[:, [1, 2, 0]]) # shape (n_tri, 3, 2)
        ke_matrix = np.empty((len(coord), 3, 3))
        area = np.abs(0.5 * self.det2x2(s[:, 0], s[:, 1]))
        #print(type(s))
        #print(type(s.T))
        ke_matrix[:] = np.einsum('ijk,kli->ijl', s, s.T) / (4. * area[:, None, None])
        # A = s
        # B = s.T
        # i,j,k,l = A.shape
        # A = np.reshape(A,(j,k*l*i))
        # B = np.reshape(B,(1,k*l*i))
        # C = np.sum(A*B,axis=2)
        #casting = 'same_kind'
        #C = np.einsum('ijk,kli->ijl', s, s.T,casting = 'same_kind', dtype = np.core.core.ndarray)
        #ke_matrix[:] = C/(4. * area[:, None, None])
        
        return ke_matrix

    def det2x2(self, s1, s2):
        """Calculate the determinant of a 2x2 matrix"""
        return s1[:, 0]*s2[:, 1] - s1[:, 1]*s2[:, 0]

    def shapeFunctionParameters(self):
        '''
        return arrays of parameters for all shape functions in all triangles on electrodes - shape ((n_el * n_per_el - 1), 3, 3)
        '''
        #print(self.tri[twoFromElectrode][isValid])
        pointsTri = self.pts[self.tri[self.twoFromElectrode][self.isValid]] # shape ((n_el * n_per_el - 1), 3, 2)
        params = np.empty((pointsTri.shape[0], 3, 3))
        params[:, :, 0] = np.multiply(pointsTri[:, [1, 2, 0], 0], pointsTri[:, [2, 0, 1], 1]) - np.multiply(pointsTri[:, [2, 0, 1], 0], pointsTri[:, [1, 2, 0], 1])
        params[:, :, 1] = pointsTri[:, [1, 2, 0], 1] - pointsTri[:, [2, 0, 1], 1]
        params[:, :, 2] = - (pointsTri[:, [1, 2, 0], 0] - pointsTri[:, [2, 0, 1], 0])
        
        return params

    def findTrianglesOnElectrodes(self):
        twoFromElectrode = (np.sum(self.tri < self.ne * self.n_per_el, axis = 1) == 2)
        nodeisElectrode = self.tri[twoFromElectrode][self.tri[twoFromElectrode] < self.ne * self.n_per_el].reshape(self.tri[twoFromElectrode].shape[0], 2)
        isValid = ( nodeisElectrode[:, 0]//self.n_per_el - nodeisElectrode[:, 1]//self.n_per_el ) == 0
        return twoFromElectrode,nodeisElectrode, isValid
    
    def admittanceMatrixC2(self):
        '''
        compute matrix to calculate integral of two shape functions
        over the length of the electrode (assuming they are non-zero) - shape ((n_el * n_per_el - 1), 3, 3, 3)
        '''
        shapeParams = self.shapeFunctionParameters()
        whereIsZero = (np.absolute(shapeParams) - 1e-12 < 0)
        indexZero = np.where(whereIsZero)
        isConst = indexZero[2] == 2 # 1 for const x, 0 for const y
        zeroShapeFunc = np.array(indexZero[1])
        indicesShapeFunctions = np.outer(np.ones(shapeParams.shape[0]), np.arange(3))
        indicesShapeFunctions[:, ~zeroShapeFunc] = 0
        #print(indexZero)
        outerOfShapeFunc = np.einsum('ijk, ipq -> ijpkq', shapeParams, shapeParams)
        #print(outerOfShapeFunc[0,0,0])
        integratingMatrix = np.empty((outerOfShapeFunc.shape[0], outerOfShapeFunc.shape[3], outerOfShapeFunc.shape[4]))
        
        '''
        for i in range(20):
            #print(self.pts[nodeisElectrode[isValid], :][i])
            print(nodeisElectrode[isValid][i])
            print(self.tri[twoFromElectrode][isValid][i])
        #print(nodeisElectrode[isValid])'''
        sortedElNodeIndices = np.sort(self.nodeisElectrode[self.isValid], axis=1)
        #print(sortedElNodeIndices)
        firstOrderY = np.empty((outerOfShapeFunc.shape[0]))
        secondOrderY = np.empty((outerOfShapeFunc.shape[0]))
        thirdOrderY = np.empty((outerOfShapeFunc.shape[0]))
        constX = np.ones((outerOfShapeFunc.shape[0], 3))
        firstOrderY[:] = self.pts[sortedElNodeIndices, :][:, 1, 1] - self.pts[sortedElNodeIndices, :][:, 0, 1] # y2 - y1
        secondOrderY[:] = 0.5 * (np.power(self.pts[sortedElNodeIndices, :][:, 1, 1], 2) - np.power(self.pts[sortedElNodeIndices, :][:, 0, 1], 2)) # 1/2 (y2^2 - y1^2)
        thirdOrderY[:] = 1./3. * (np.power(self.pts[sortedElNodeIndices, :][:, 1, 1], 3) - np.power(self.pts[sortedElNodeIndices, :][:, 0, 1], 3)) # 1/3 (y2^3 - y1^3)
        constX[:, 1] = self.pts[sortedElNodeIndices, :][:, 1, 0]
        constX = np.einsum('ij, ik -> ijk', constX, constX)
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
        #intm = np.array(integratingMatrix)
        #print(constX)
        firstOrderX = np.empty((outerOfShapeFunc.shape[0]))
        secondOrderX = np.empty((outerOfShapeFunc.shape[0]))
        thirdOrderX = np.empty((outerOfShapeFunc.shape[0]))
        constY = np.ones((outerOfShapeFunc.shape[0], 3))
        firstOrderX[:] = self.pts[sortedElNodeIndices, :][:, 1, 0] - self.pts[sortedElNodeIndices, :][:, 0, 0] # x2 - x1
        secondOrderX[:] = 0.5 * (np.power(self.pts[sortedElNodeIndices, :][:, 1, 0], 2) - np.power(self.pts[sortedElNodeIndices, :][:, 0, 0], 2)) # 1/2 (x2^2 - x1^2)
        thirdOrderX[:] = 1./3. * (np.power(self.pts[sortedElNodeIndices, :][:, 1, 0], 3) - np.power(self.pts[sortedElNodeIndices, :][:, 0, 0], 3)) # 1/3 (x2^3 - x1^3)
        constY[:, 2] = self.pts[sortedElNodeIndices, :][:, 1, 1]
        constY = np.einsum('ij, ik -> ijk', constY, constY)
        #print(constY)
        indicesConstX = np.where(isConst)[0]
        indicesConstY = np.where(~isConst)[0]
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
        integratingMatrix[indicesConstX] = np.multiply(integratingMatrix[indicesConstX], constX[indicesConstX])
        integratingMatrix[indicesConstY] = np.multiply(integratingMatrix[indicesConstY], constY[indicesConstY])

        admittanceMatrix = np.einsum('ijklm, ilm -> ijk', outerOfShapeFunc, integratingMatrix)
        
        admittanceMatrix[:] = np.abs(admittanceMatrix)
        admittanceMatrix[admittanceMatrix < 1e-18] = 0
        #admittanceMatrix2 = np.sum(np.multiply(outerOfShapeFunc, integratingMatrix[:, None, None, :, :]), axis = [3,4])

        #print(admittanceMatrix[:50,:50])
        #number_of_equal = np.sum(np.equal(np.round_(admittanceMatrix, 16), np.round_(admittanceMatrix2, 16)))
        #print(number_of_equal)
        #print(number_of_equal == admittanceMatrix.shape[0] * admittanceMatrix.shape[1] * admittanceMatrix.shape[2])
        return admittanceMatrix

    def admittanceMatrixE(self):
        shapeParams = self.shapeFunctionParameters()
        whereIsZero = (np.abs(shapeParams) - 1e-12 < 0)
        indexZero = np.where(whereIsZero)
        isConst = indexZero[2] == 2 # 1 for const x, 0 for const y
        indicesConstX = np.where(isConst)[0]
        indicesConstY = np.where(~isConst)[0]
        sortedElNodeIndices = np.sort(self.nodeisElectrode[self.isValid], axis=1)
        admittanceMatrixE = np.zeros((self.n_pts, self.ne))
        shapeMatrix = np.zeros((shapeParams.shape[0], shapeParams.shape[1], 2))
        integratingMatrix = np.zeros((shapeParams.shape[0], 2))
        shapeMatrix[indicesConstY, :, 0] = shapeParams[indicesConstY, :, 0] + shapeParams[indicesConstY, :, 2] * self.pts[sortedElNodeIndices, :][indicesConstY, 1, 1][:, None]
        shapeMatrix[indicesConstY, :, 1] = shapeParams[indicesConstY, :, 1]
        shapeMatrix[indicesConstX, :, 0] = shapeParams[indicesConstX, :, 0] + shapeParams[indicesConstX, :, 1] * self.pts[sortedElNodeIndices, :][indicesConstX, 1, 0][:, None]
        shapeMatrix[indicesConstX, :, 1] = shapeParams[indicesConstX, :, 2]
        integratingMatrix[indicesConstY, 0] = self.pts[sortedElNodeIndices, :][indicesConstY, 1, 0] - self.pts[sortedElNodeIndices, :][indicesConstY, 0, 0]
        integratingMatrix[indicesConstY, 1] = 0.5 * (np.power(self.pts[sortedElNodeIndices, :][indicesConstY, 1, 0], 2) - np.power(self.pts[sortedElNodeIndices, :][indicesConstY, 0, 0], 2))
        integratingMatrix[indicesConstX, 0] = self.pts[sortedElNodeIndices, :][indicesConstX, 1, 1] - self.pts[sortedElNodeIndices, :][indicesConstX, 0, 1]
        integratingMatrix[indicesConstX, 1] = 0.5 * (np.power(self.pts[sortedElNodeIndices, :][indicesConstX, 1, 1], 2) - np.power(self.pts[sortedElNodeIndices, :][indicesConstX, 0, 1], 2))
        #print(integratingMatrix.shape)
        integrals = np.einsum('ijk, ik -> ij', shapeMatrix, integratingMatrix)
        integrals[:] = np.abs(integrals)
        #integr = np.sum(np.multiply(shapeMatrix, integratingMatrix[:, None]), axis=2)

        #print(np.sum(np.round_(integrals, 16) == np.round_(integr, 16)))

        indexElectrode = sortedElNodeIndices[:, 0] // self.n_per_el
        #print(indexElectrode)
        integrals = - integrals / self.z[indexElectrode][:, None, None]
        integrals = integrals.ravel()
        indexElectrode = np.tile(indexElectrode, (self.n_per_el, 1)).T.ravel()
        #print(self.tri[twoFromElectrode][isValid])
        indexNode = self.tri[self.twoFromElectrode][self.isValid].ravel()
        
        #admittanceMatrixE [self.tri[twoFromElectrode][isValid].ravel(), indexElectrode] += integrals.ravel()
        indSort = np.argsort(indexNode)
        indexNode = indexNode[indSort]
        indexElectrode = indexElectrode[indSort]
        integrals = integrals[indSort]

        unique, counts = np.unique(indexNode, return_counts=True)
        #print("number of unique entries", unique.shape)
        #print("counts \n", counts)
        index_pointer = np.zeros(self.n_pts + 1)
        sum_count = np.cumsum(counts)
        #print(sum_count)
        index_pointer[unique[:]+1] = sum_count[:]
        #print(index_pointer)
        nonzeroes = np.nonzero(index_pointer)[0]
        #print(nonzeroes)
        mask = np.zeros(index_pointer.shape[0], dtype='b1')
        mask[nonzeroes] = True
        mask[0] = True
        zeroes = np.where(~mask)[0]

        #time_loop = time()
        while (index_pointer[1:] == 0).any():
            index_pointer[zeroes] = index_pointer[zeroes - 1]
        '''for i in range(index_pointer.shape[0]):
            if i == 0:
                continue
            elif index_pointer[i] == 0:
                index_pointer[i] = index_pointer[i-1]'''
        #print('time for loop ',time()-time_loop)
        index_pointer2 = np.arange(self.n_pts + 1)
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
        #print('sum zeroes ',np.sum(adm>0))
        return adm; 

    def admittanceMatrixD(self):
        all_el_nodes_coords = self.pts[:(self.ne * self.n_per_el)].reshape((self.ne, self.n_per_el, 2))
        lengths = np.linalg.norm((all_el_nodes_coords[:, 0] - all_el_nodes_coords[:, (self.n_per_el - 1)]), axis=1)
        admittanceMatrixD = np.diag(lengths/self.z)
        return admittanceMatrixD


def volt_mat(ne):
    #generates all possible pairs of voltage measurements
    step_arr = np.arange(ne)
    parser = None
    A = np.arange(ne)
    M = np.dot(np.ones(ne)[:,None], A[None, :]) % ne
    N = (M + step_arr[:, None]) % ne    
    pair_mat = np.stack((N, M), axis=-1)
    pair_mat = pair_mat.reshape(((ne)**2, 2))

    ind = pair_mat[:,0] < pair_mat[:, 1]

    return pair_mat[ind]

def smear(f, fb, pairs):
    """
    build smear matrix B for bp

    Parameters
    ----------
    f : NDArray
        potential on nodes
    fb : NDArray
        potential on adjacent electrodes
    pairs : NDArray
        electrodes numbering pairs

    Returns
    -------
    NDArray
        back-projection matrix
    """
    #b_matrix = np.empty(size=(len(pairs), len(f)))
    #t1 = time()
    #b_matrix = []
    f = np.array(f)
    fb = np.array(fb)
    pairs= np.array(pairs)
    i = np.arange(len(pairs))
    min_fb = np.amin(fb[pairs], axis=1)
    max_fb = np.amax(fb[pairs], axis=1)
    b_matrix = np.empty((len(pairs), len(f)))
    #index[i, :] = (min_fb[i] < f.all()) & (f.all() <= max_fb[i])
    b_matrix[:] = (min_fb[i, None] < f[None]) & (f[None] <= max_fb[i, None])
    #t2 = time()
    '''
    for i, j in pairs:
        f_min, f_max = min(fb[i], fb[j]), max(fb[i], fb[j])
        b_matrix.append((f_min < f) & (f <= f_max))
    b_matrix = np.array(b_matrix)
    '''
    #print("matrices: ", t2 - t1)
    #print("their loop ", time() - t2)
    return np.asnumpy(b_matrix)


def subtract_row(v, pairs):
    """
    v_diff[k] = v[i, :] - v[j, :]

    Parameters
    ----------
    v : NDArray
        Nx1 boundary measurements vector or NxM matrix
    pairs : NDArray
        Nx2 subtract_row pairs

    Returns
    -------
    NDArray
        difference measurements
    """
    i = pairs[:, 0]
    j = pairs[:, 1]
    # row-wise/element-wise operation on matrix/vector v
    v_diff = v[i] - v[j]

    return v_diff
