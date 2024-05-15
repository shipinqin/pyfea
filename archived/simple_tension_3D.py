import numpy as np
import pandas as pd
from io import StringIO


def readinput(inp_fpath, from_abq=False):

    with open(inp_fpath, 'r') as f:
        content = f.read()

    nodes, elements = content.lower().split('*node\n')[1].split('*element\n')

    # nodes = np.array(lines[node_start:node_end])
    # elements = np.array(lines[element_start:])
    nodes = np.loadtxt(StringIO(nodes), delimiter=',')
    elements = np.loadtxt(StringIO(elements), delimiter=',').astype(int)

    if len(nodes.shape) == 1:
        nodes = np.array([nodes]).T
    if len(elements.shape) == 1:
        elements = np.array([elements])
    if from_abq:
        nodes = nodes[:,1:]
        elements = elements[:,1:] - 1
    nnodes, ndim = nodes.shape
    nel, nnodes_el = elements.shape

    return nodes, elements, nnodes, ndim, nel, nnodes_el


def shape_func(xi, nnodes_el, ndim):

    # It seems, the sequence of shape function needs to be in accordance with the local coords system!

    # xi (1 x ndim) is the local coordinate
    N = np.zeros(nnodes_el)
    if ndim ==3 and nnodes_el == 8:
        for n_local in range(1,9):
            T1 = 1-xi[0] if n_local in [1, 4, 5, 8] else 1+xi[0]
            T2 = 1-xi[1] if n_local in [1, 2, 5, 6] else 1+xi[1]
            T3 = 1-xi[2] if n_local in [1, 2, 3, 4] else 1+xi[2]
            N[n_local-1] = 1/8*T1*T2*T3
    elif ndim == 1 and nnodes_el == 2:
        N = np.array([(1-xi)/2, (1+xi)/2])
    else:
        raise f'Model does not support ndim={ndim} and nnodes_el={nnodes_el}'

    return N


def shape_func_deriv(xi, nnodes_el, ndim):

    # x.T*shape_dev gives dx/dxi
    dNdxi = np.zeros((nnodes_el, ndim))
    if ndim ==3 and nnodes_el == 8:
        dNdxi[0] = [-(1-xi[1])*(1-xi[2]), -(1-xi[0])*(1-xi[2]), -(1-xi[0])*(1-xi[1])]
        dNdxi[1] = [ (1-xi[1])*(1-xi[2]), -(1+xi[0])*(1-xi[2]), -(1+xi[0])*(1-xi[1])]
        dNdxi[2] = [ (1+xi[1])*(1-xi[2]),  (1+xi[0])*(1-xi[2]), -(1+xi[0])*(1+xi[1])]
        dNdxi[3] = [-(1+xi[1])*(1-xi[2]),  (1-xi[0])*(1-xi[2]), -(1-xi[0])*(1+xi[1])]
        dNdxi[4] = [-(1-xi[1])*(1+xi[2]), -(1-xi[0])*(1+xi[2]),  (1-xi[0])*(1-xi[1])]
        dNdxi[5] = [ (1-xi[1])*(1+xi[2]), -(1+xi[0])*(1+xi[2]),  (1+xi[0])*(1-xi[1])]
        dNdxi[6] = [ (1+xi[1])*(1+xi[2]),  (1+xi[0])*(1+xi[2]),  (1+xi[0])*(1+xi[1])]
        dNdxi[7] = [-(1+xi[1])*(1+xi[2]),  (1-xi[0])*(1+xi[2]),  (1-xi[0])*(1+xi[1])]
        dNdxi = 1/8*dNdxi
    elif ndim == 1 and nnodes_el == 2:
        dNdxi = np.array([[-1/2], [1/2]])
    else:
        raise f'Model does not support ndim={ndim} and nnodes_el={nnodes_el}'

    return dNdxi


def mat_stiffness(eps_, nnodes_el, ndim, E=201000, nu=0.3):

    if ndim ==3:
        mu = E/2*(1+nu)  # shear modulus
        C = np.zeros((3, 3, 3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        if i==k and j==l:
                            C[i,j,k,l] = C[i,j,k,l] + mu
                        if i==j and k==l:
                            C[i,j,k,l] = C[i,j,k,l] + 2*mu*nu/(1-2*nu)
                        if i==l and j==k:
                            C[i,j,k,l] = C[i,j,k,l] + mu
    elif ndim == 1:
        C = E

    dsde = C
    return dsde


def integration_points_weights(nnodes_el, ndim, reduced=False):

    if ndim ==3 and nnodes_el == 8:
        if reduced:
            xi = np.array([0, 0, 0])
            w  = np.array([2])
        else:
            xi = np.zeros((nnodes_el, ndim))
            sqrt3inv = 1/np.sqrt(3)
            x1D = [-sqrt3inv, sqrt3inv]
            for k in range(2):
                for j in range(2):
                    for i in range(2):
                        n = 4*(k-1) + 2*(j-1) + i - 1
                        xi[n] = [x1D[i], x1D[j], x1D[k]]
            w = np.array([1.,1.,1.,1.,1.,1.,1.,1.])
    elif ndim == 1:
        xi = np.array([0])
        w  = np.array([2])

    return xi, w


def element_stiffness(nodes, element, displacements, nnodes_el, ndim):

    xi_list, w = integration_points_weights(nnodes_el, ndim, reduced=False)
    K_e = np.zeros((ndim*nnodes_el, ndim*nnodes_el))

    node_num = element
    xs_node = nodes[node_num]                # (nnodes_el x ndim)
    for ii, xi in enumerate(xi_list):
        # This part are performed at the integration point
        # xs_int  = xs_node.T@shape_func(xi, nnodes_el, ndim)    # (ndim x 1) integration point coords
        dNdxi = shape_func_deriv(xi, nnodes_el, ndim)            # (nnodes_el x ndim)
        dxdxi = xs_node.T@dNdxi               # (ndim x ndim)
        dxidx = np.linalg.inv(dxdxi)          # (ndim x ndim)
        dNdx  = dNdxi@dxidx                   # (nnodes_el x ndim)
        J = np.linalg.det(dxdxi)

        eps_ = 1/2*(displacements.T@dNdx + dNdx.T@displacements)  # This line projects displacement at the nodes to the strain at the integration point
        dsde = mat_stiffness(eps_, nnodes_el, ndim)

        for A in range(nnodes_el):
            for i in range(ndim):
                for B in range(nnodes_el):
                    for j in range(ndim):
                        for k in range(ndim):
                            for l in range(ndim):
                                K_e[ndim*A+i, ndim*B+k] += w[ii] * J * dsde[i,j,k,l] * (dNdx[A,j]*dNdx[B,l])  # (ndim*nnodes_el x ndim*nnodes_el)

    return K_e


def global_stiffness(nodes, elements, displacements, nnodes, ndim):

    K_global = np.zeros((ndim*nnodes, ndim*nnodes))
    for element in elements:
        K_e = element_stiffness(nodes, element, displacements[element], nnodes_el, ndim)
        node_num = element
        for i, node1 in enumerate(node_num):
            for dim1 in range(ndim):
                for j,node2 in enumerate(node_num):
                    for dim2 in range(ndim):
                        K_global[ndim*node1+dim1, ndim*node2+dim2] += K_e[ndim*i+dim1, ndim*j+dim2]

    return K_global


if __name__ == '__main__':
    nodes, elements, nnodes, ndim, nel, nnodes_el = readinput('L-tension_mesh.inp', from_abq=True)
    # BC = [node number, dof, displacement]
    # BCs = np.array([[ 0, 0, 0],
    #                 [ 0, 2, 0],
    #                 [ 1, 2, 0],
    #                 [ 0, 1, 0],
    #                 [ 1, 1, 0],
    #                 [ 4, 1, 0],
    #                 [ 5, 1, 0],
    #                 [ 2, 1, 0.1],
    #                 [ 3, 1, 0.1],
    #                 [ 6, 1, 0.1],
    #                 [ 7, 1, 0.1]])
    # BCs = np.array([[ 0, 0, 0],
    #                 [ 0, 2, 0],
    #                 [ 1, 2, 0],
    #                 [ 0, 1, 0],
    #                 [ 1, 1, 0],
    #                 [ 4, 1, 0],
    #                 [ 5, 1, 0],
    #                 [ 8, 1, 0.1],
    #                 [ 9, 1, 0.1],
    #                 [10, 1, 0.1],
    #                 [11, 1, 0.1]])
    bottom = [1,  2,  8,  9, 10, 16]
    top    = [4,  5, 12, 13]
    corner = [2]
    right_corner = [1]
    BCs = []
    for n in corner:
        BCs.append([n, 0, 0])
        BCs.append([n, 2, 0])
    for n in bottom:
        BCs.append([n, 1, 0])
    for n in top:
        BCs.append([n, 1, 0.1])
    for n in right_corner:
        BCs.append([n, 2, 0])

    BCs = np.array(BCs)
    BCs[:,0] = BCs[:,0]-1

    displacement = np.zeros(ndim*nnodes)
    K = global_stiffness(nodes, elements, displacement, nnodes, ndim)
    F = np.zeros((ndim*nnodes))
    ind = []
    K_mod = K
    for BC in BCs:
        ind.append(int(ndim*BC[0]+BC[1]))
    for i, val in enumerate(ind):
        F -= K[val] * BCs[i,2]
        F[val] = BCs[i,2]
        K_mod[:,val], K[val,:] = 0, 0
        K_mod[val,val] = 1
    # K = np.delete(K, ind, axis=0)
    # K = np.delete(K, ind, axis=1)

    print(K_mod.shape)
    U = np.linalg.inv(K_mod)@F
    F = K@U

    U = pd.DataFrame(U.reshape((nnodes,ndim)), columns=['U1','U2','U3'])
    F = pd.DataFrame(F.reshape((nnodes,ndim)), columns=['F1','F2','F3'])
    print(U)
    pass
