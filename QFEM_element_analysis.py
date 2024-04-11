import numpy as np
import matplotlib.pyplot as plt


def shape_func(el_type, xi, nnodes_el):
    # Shape function sequence should be in accordance with the node sequence in element connectivity!

    # xi (1 x ndim) is the local coordinate
    N = np.zeros(nnodes_el)
    if 'C3D8' in el_type:
        for n_local in range(1,9):
            T1 = 1-xi[0] if n_local in [1, 4, 5, 8] else 1+xi[0]
            T2 = 1-xi[1] if n_local in [1, 2, 5, 6] else 1+xi[1]
            T3 = 1-xi[2] if n_local in [1, 2, 3, 4] else 1+xi[2]
            N[n_local-1] = 1/8*T1*T2*T3
    elif 'C1D2' in el_type:
        N = np.array([(1-xi)/2, (1+xi)/2])
    elif 'C1D3' in el_type:
        N = np.array([-xi*(1-xi)/2, xi*(1+xi)/2, (1-xi)*(1+xi)])
    elif 'C2D3' in el_type:
        N[0] = xi[0]
        N[1] = xi[1]
        N[2] = 1.-xi[0]-xi[1]
    elif 'C2D6' in el_type:
        xi2 = 1.-xi[0]-xi[1]
        N[0] = (2.*xi[0]-1.)*xi[0]
        N[1] = (2.*xi[1]-1.)*xi[1]
        N[2] = (2.*xi2-1.)*xi2
        N[3] = 4.*xi[0]*xi[1]
        N[4] = 4.*xi[1]*xi2
        N[5] = 4.*xi2*xi[0]
    else:
        raise f'Element type {el_type} is not supported yet'

    return np.array(N)


def shape_func_dev(el_type, xi, nnodes_el, ndim):
    # Shape function sequence should be in accordance with the node sequence in element connectivity!

    # xi (1 x ndim) is the local coordinate
    # dNdxi is (nnodes_el x ndim)
    dNdxi = np.zeros((nnodes_el, ndim))
    if 'C3D8' in el_type:
        dNdxi[0] = [-(1-xi[1])*(1-xi[2]), -(1-xi[0])*(1-xi[2]), -(1-xi[0])*(1-xi[1])]
        dNdxi[1] = [ (1-xi[1])*(1-xi[2]), -(1+xi[0])*(1-xi[2]), -(1+xi[0])*(1-xi[1])]
        dNdxi[2] = [ (1+xi[1])*(1-xi[2]),  (1+xi[0])*(1-xi[2]), -(1+xi[0])*(1+xi[1])]
        dNdxi[3] = [-(1+xi[1])*(1-xi[2]),  (1-xi[0])*(1-xi[2]), -(1-xi[0])*(1+xi[1])]
        dNdxi[4] = [-(1-xi[1])*(1+xi[2]), -(1-xi[0])*(1+xi[2]),  (1-xi[0])*(1-xi[1])]
        dNdxi[5] = [ (1-xi[1])*(1+xi[2]), -(1+xi[0])*(1+xi[2]),  (1+xi[0])*(1-xi[1])]
        dNdxi[6] = [ (1+xi[1])*(1+xi[2]),  (1+xi[0])*(1+xi[2]),  (1+xi[0])*(1+xi[1])]
        dNdxi[7] = [-(1+xi[1])*(1+xi[2]),  (1-xi[0])*(1+xi[2]),  (1-xi[0])*(1+xi[1])]
        dNdxi = 1/8*dNdxi
    elif 'C1D2' in el_type:
        dNdxi = np.array([[-1/2], [1/2]])
    elif 'C1D3' in el_type:
        dNdxi = np.array([xi-1/2, xi+1/2, -2*xi])
    elif 'C2D3' in el_type:
        dNdxi[0,0] = 1.
        dNdxi[1,1] = 1.
        dNdxi[2,0] = -1.
        dNdxi[2,1] = -1.
    # elif 'C2D6' in el_type:

    else:
        raise f'Element type {el_type} is not supported yet'

    return np.array(N)


if __name__ == '__main__':
