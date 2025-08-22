import numpy as np
from typing import Sequence

def get_shape_func(nelnodes: int, ndim: int) -> Sequence[callable]:
    '''
    nelnodes: Number of element nodes
    ndim: Number of dimensions

    Returns a sequence of shape functions (nelnodes, ) that accepts local coordinates xi
    '''

    # ------------------------------
    # 1D elements
    if ndim == 1:
        if nelnodes == 2:  # C1D2
            N = lambda xi: np.array([(1-xi)/2, (1+xi)/2])
        elif nelnodes == 3:  # C1D3
            N = lambda xi: np.array([-xi*(1-xi)/2, xi*(1+xi)/2, (1-xi)*(1+xi)])

    # ------------------------------
    # 2D elements
    elif ndim == 2:
        if nelnodes == 3:  # 1st order triangle
            N = lambda xi:np.array([xi[0], xi[1], 1.-xi[0]-xi[1]])
        elif nelnodes == 6:  # 2nd order triangle
            xi2 = lambda xi: 1-xi[0]-xi[1]
            N = lambda xi: np.array([(2*xi[0]-1)*xi[0],
                                       (2*xi[1]-1)*xi[1],
                                       (2*xi2(xi)-1)*xi2(xi),
                                       4*xi[0]*xi[1],
                                       4*xi[1]*xi2(xi),
                                       4*xi[0]*xi2(xi)])

        elif nelnodes == 4:  # 1st order quad
            N = lambda xi: np.array([(1-xi[0])*(1-xi[1])/4,
                                       (1+xi[0])*(1-xi[1])/4,
                                       (1+xi[0])*(1+xi[1])/4,
                                       (1-xi[0])*(1+xi[1])/4])
        elif nelnodes == 8:  # 2nd order quad
            N = lambda xi: np.array([-(1-xi[0])*(1-xi[1])*(1+xi[0]+xi[1])/4,
                                       (1+xi[0])*(1-xi[1])*(xi[0]-xi[1]-1)/4,
                                       (1+xi[0])*(1+xi[1])*(xi[0]+xi[1]-1)/4,
                                       (1-xi[0])*(1+xi[1])*(xi[0]-xi[1]-1)/4,
                                       (1-xi[0]*xi[0])*(1-xi[1])/2,
                                       (1+xi[0])*(1-xi[1]*xi[1])/2,
                                       (1-xi[0]*xi[0])*(1+xi[1])/2,
                                       (1-xi[0])*(1-xi[1]*xi[1])/2])

    # ------------------------------
    # 3D elements
    elif ndim == 3:
        N_temp = []
        if nelnodes == 8:  # C3D8
            for n_local in range(1,9):
                T1 = lambda xi: 1-xi[0] if n_local in [1, 4, 5, 8] else 1+xi[0]
                T2 = lambda xi: 1-xi[1] if n_local in [1, 2, 5, 6] else 1+xi[1]
                T3 = lambda xi: 1-xi[2] if n_local in [1, 2, 3, 4] else 1+xi[2]
                N_temp.append(lambda xi: 1/8*T1(xi)*T2(xi)*T3(xi))
            N = lambda xi: np.array([Ni(xi) for Ni in N_temp])
        else:
            raise NotImplementedError(f"Model does not support {ndim=} and {nelnodes=}")

    return N

def get_shape_func_deriv(nelnodes: int, ndim: int) -> Sequence[callable]:

    '''
    nelnodes: Number of element nodes
    ndim: Number of dimensions

    Returns a sequence of shape function derivatives (nelnodes, ndim) that accepts local coordinates xi
    '''

    # Shape function sequence should be in accordance with the node sequence in element connectivity!

    # x.T*shape_dev gives dx/dxi
    dNdxi = np.zeros((nelnodes, ndim))

    # ------------------------------
    # 1D elements
    if ndim == 1:
        if nelnodes == 2:  # C1D2
            dNdxi = lambda xi: np.array([[-1/2], [1/2]])
        elif nelnodes == 3:  # C1D3
            dNdxi = lambda xi: np.array([xi-1/2, xi+1/2, -2*xi])

    # ------------------------------
    # 2D elements
    elif ndim == 2:
        if nelnodes == 3:  # 1st order triangle
            dNdxi = lambda xi: np.array([[1, 0],
                                         [0, 1],
                                         [-1, -1]])

        elif nelnodes == 6:  # 2nd order triangle
            xi2 = lambda xi: 1-xi[0]-xi[1]
            dNdxi = lambda xi: np.array([[4*xi[0]-1, 0],
                                         [0, 4*xi[1]-1],
                                         [-(4*xi2(xi)-1), -(4*xi2(xi)-1)],
                                         [4*xi[1], 4*xi[0]],
                                         [-4*xi[1], 4*(xi2(xi)-xi[1])],
                                         [4*(xi2(xi)-xi[0]), -4*xi[0]]])
            # In Bower's code, dNdxi[4] and dNdxi[5] are wrong
        elif nelnodes == 4:  # 1st order quad
            dNdxi = lambda xi: np.array([ [-(1-xi[1])/4, -(1-xi[0])/4],
                                          [(1-xi[1])/4, -(1+xi[0])/4],
                                          [(1+xi[1])/4, (1+xi[0])/4],
                                          [-(1+xi[1])/4, (1-xi[0])/4]])
        elif nelnodes == 8:  # 2nd order quad
            dNdxi = lambda xi: np.array([[(1-xi[1]*(2*xi[0]+xi[1]))/4, (1-xi[0]*(xi[0]+2*xi[1]))/4],
                                            [(1-xi[1]*(2*xi[0]-xi[1]))/4, (1+xi[0]*(-xi[0]+2*xi[1]))/4],
                                            [(1+xi[1]*(2*xi[0]+xi[1]))/4, (1+xi[0]*(xi[0]+2*xi[1]))/4],
                                            [(1+xi[1]*(2*xi[0]-xi[1]))/4, (1-xi[0]*(-xi[0]+2*xi[1]))/4],
                                            [-xi[0]*(1-xi[1]), -(1-xi[0]*xi[0])/2],
                                            [(1-xi[1]*xi[1])/2, -(1+xi[0])*xi[1]],
                                            [-xi[0]*(1+xi[1]), (1-xi[0]*xi[0])/2],
                                            [-(1-xi[1]*xi[1])/2, -(1-xi[0])*xi[1]]])

    # ------------------------------
    # 3D elements
    elif ndim ==3:
        if nelnodes == 8:  # C3D8
            dNdxi = lambda xi: 1/8*np.array([[-(1-xi[1])*(1-xi[2]), -(1-xi[0])*(1-xi[2]), -(1-xi[0])*(1-xi[1])],
                                            [ (1-xi[1])*(1-xi[2]), -(1+xi[0])*(1-xi[2]), -(1+xi[0])*(1-xi[1])],
                                            [ (1+xi[1])*(1-xi[2]),  (1+xi[0])*(1-xi[2]), -(1+xi[0])*(1+xi[1])],
                                            [-(1+xi[1])*(1-xi[2]),  (1-xi[0])*(1-xi[2]), -(1-xi[0])*(1+xi[1])],
                                            [-(1-xi[1])*(1+xi[2]), -(1-xi[0])*(1+xi[2]),  (1-xi[0])*(1-xi[1])],
                                            [ (1-xi[1])*(1+xi[2]), -(1+xi[0])*(1+xi[2]),  (1+xi[0])*(1-xi[1])],
                                            [ (1+xi[1])*(1+xi[2]),  (1+xi[0])*(1+xi[2]),  (1+xi[0])*(1+xi[1])],
                                            [-(1+xi[1])*(1+xi[2]),  (1-xi[0])*(1+xi[2]),  (1-xi[0])*(1+xi[1])]])

    else:
        raise f'Model does not support ndim={ndim} and nelnodes={nelnodes}'

    return dNdxi


def get_integration_points_weights(nelnodes: int, ndim: int, reduced: bool = False) -> tuple[np.ndarray, np.ndarray]:

    x = 0.5773502692
    # Integration points sequence should be in accordance with the node sequence in element connectivity!
    if ndim == 1:  # 1D model
        if reduced:
            xi = np.array([[0.]])
            w  = np.array([2.])
        else:
            xi = np.array([[-x], [x]])
            w  = np.array([1., 1.])

    elif ndim == 2: # 2D model
        if nelnodes in [3, 6]:  # triangle element
            if reduced:
                xi = np.array([[1/3, 1/3]])
                w = np.array([1/2])
            else:
                xi = np.array([[0.6, 0.2], [0.2, 0.6], [0.2, 0.2]])
                w  = np.array([1/6, 1/6, 1/6])
        elif nelnodes in [4, 8]:  # quadrilateral element
            if reduced:
                xi = np.array([[0., 0.]])
                w = np.array([4.])
            else:
                xi = np.array([[-x, -x], [x, -x], [-x, x], [x, x]])
                w = np.array([1., 1., 1., 1.])

    elif ndim ==3:  # 3D model
        if nelnodes in [4, 10]:  # tetrahedral element
            if reduced:
                xi = np.array([[1/4, 1/4, 1/4]])
                w = np.array([1/6])
            else:
                x1, x2 = 0.58541020, 0.13819660
                xi = np.array([[x1, x2, x2], [x2, x1, x2], [x2, x2, x1], [x2, x2, x2]])
                w  = np.array([1/24, 1/24, 1/24, 1/24])
        if nelnodes in [8, 20]:  # hex element
            if reduced:
                xi = np.array([[0., 0., 0.]])
                w  = np.array([8.])
            else:
                n_int = 8
                xi = np.zeros((n_int, ndim))
                sqrt3inv = 1/np.sqrt(3)
                x1D = [-sqrt3inv, sqrt3inv]
                for k in range(2):
                    for j in range(2):
                        for i in range(2):
                            n = 4*(k-1) + 2*(j-1) + i - 1
                            xi[n] = [x1D[i], x1D[j], x1D[k]]
                w = np.array([1.,1.,1.,1.,1.,1.,1.,1.])

    return xi, w

