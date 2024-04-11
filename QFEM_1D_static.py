import numpy as np
from io import StringIO


class QFEM():
    def __init__(self, inp_fpath):

        with open(inp_fpath, 'r') as f:
            content = f.read()

        nodes, elements = content.lower().split('*node\n')[1].split('*element\n')

        # nodes = np.array(lines[node_start:node_end])
        # elements = np.array(lines[element_start:])
        nodes = np.loadtxt(StringIO(nodes), delimiter=',')
        elements = np.loadtxt(StringIO(elements), delimiter=',').astype(int)
        elements -= 1  # Python numbering starts from zero

        if len(nodes.shape) == 1:
            nodes = np.array([nodes]).T
        self.nnodes, self.ndim = nodes.shape
        if len(elements.shape) == 1:
            elements = np.array([elements])
        self.nel, self.nnodes_el = elements.shape

        self.nodes, self.elements = nodes, elements

        self.nint =2
        self.nshape = 3

        # Disp boundary conditions
        self.BCs = np.array([[0, 0, 0]])   # BC = [node number, dof, displacement]

        # Traction
        traction = 2
        self.traction = np.array([[self.nodes[-1], 'x', 2]])

        # Body force
        self.body_force = ['x', 10]

        A = 1
        mu = 50
        nu = 0.3
        const = 2*mu*A*(1-nu)/(1-2*nu)
        self.E = const


def shape_func(xi, nnodes_el, ndim):

    # Shape function sequence should be in accordance with the node sequence in element connectivity!

    # xi (1 x ndim) is the local coordinate
    N = np.zeros(nnodes_el)
    if ndim ==3 and nnodes_el == 8:  # C3D8
        for n_local in range(1,9):
            T1 = 1-xi[0] if n_local in [1, 4, 5, 8] else 1+xi[0]
            T2 = 1-xi[1] if n_local in [1, 2, 5, 6] else 1+xi[1]
            T3 = 1-xi[2] if n_local in [1, 2, 3, 4] else 1+xi[2]
            N[n_local-1] = 1/8*T1*T2*T3
    elif ndim == 1 and nnodes_el == 2:  # C1D2
        N = np.array([(1-xi)/2, (1+xi)/2])
    elif ndim == 1 and nnodes_el == 2:  # C1D3
        N = np.array([-xi*(1-xi)/2, xi*(1+xi)/2, (1-xi)*(1+xi)])
    else:
        raise f'Model does not support ndim={ndim} and nnodes_el={nnodes_el}'

    return N


def shape_func_dev(xi, nnodes_el, ndim):

    # Shape function sequence should be in accordance with the node sequence in element connectivity!

    # x.T*shape_dev gives dx/dxi
    dNdxi = np.zeros((nnodes_el, ndim))
    if ndim ==3 and nnodes_el == 8:  # C3D8
        dNdxi[0] = [-(1-xi[1])*(1-xi[2]), -(1-xi[0])*(1-xi[2]), -(1-xi[0])*(1-xi[1])]
        dNdxi[1] = [ (1-xi[1])*(1-xi[2]), -(1+xi[0])*(1-xi[2]), -(1+xi[0])*(1-xi[1])]
        dNdxi[2] = [ (1+xi[1])*(1-xi[2]),  (1+xi[0])*(1-xi[2]), -(1+xi[0])*(1+xi[1])]
        dNdxi[3] = [-(1+xi[1])*(1-xi[2]),  (1-xi[0])*(1-xi[2]), -(1-xi[0])*(1+xi[1])]
        dNdxi[4] = [-(1-xi[1])*(1+xi[2]), -(1-xi[0])*(1+xi[2]),  (1-xi[0])*(1-xi[1])]
        dNdxi[5] = [ (1-xi[1])*(1+xi[2]), -(1+xi[0])*(1+xi[2]),  (1+xi[0])*(1-xi[1])]
        dNdxi[6] = [ (1+xi[1])*(1+xi[2]),  (1+xi[0])*(1+xi[2]),  (1+xi[0])*(1+xi[1])]
        dNdxi[7] = [-(1+xi[1])*(1+xi[2]),  (1-xi[0])*(1+xi[2]),  (1-xi[0])*(1+xi[1])]
        dNdxi = 1/8*dNdxi
    elif ndim == 1 and nnodes_el == 2:  # C1D2
        dNdxi = np.array([[-1/2], [1/2]])
    elif ndim == 1 and nnodes_el == 3:  # C1D3
        dNdxi = np.array([xi-1/2, xi+1/2, -2*xi])
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

    # Integration points sequence should be in accordance with the node sequence in element connectivity!

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
    elif ndim == 1 and nnodes_el==2:
        xi = np.array([0])
        w  = np.array([2])
    elif ndim == 1 and nnodes_el==3:
        xi = np.array([[-0.5773502692], [0.5773502692]])
        w  = np.array([[1], [1]])

    return xi, w


def element_stiffness(model):

    nnodes_el, ndim = model.nnodes_el, model.ndim
    xi_list, w = integration_points_weights(nnodes_el, ndim, reduced=False)
    K_e = np.zeros((model.nel, nnodes_el*ndim, nnodes_el*ndim))

    for iel, element in enumerate(model.elements):
        node_num = element
        xs_node = model.nodes[node_num]                # (nnodes_el x ndim)
        for ixi, xi in enumerate(xi_list):
            # This part are performed at the integration point
            # xs_int  = xs_node.T@shape_func(xi, nnodes_el, ndim)    # (ndim x 1) integration point coords
            dNdxi = shape_func_dev(xi, nnodes_el, ndim)            # (nnodes_el x ndim)
            dxdxi = xs_node.T@dNdxi               # (ndim x ndim)
            # dxidx = np.linalg.inv(dxdxi)          # (ndim x ndim)
            # dNdx  = dNdxi@dxidx                   # (nnodes_el x ndim)
            J = np.linalg.det(dxdxi)
            for inode, dNdxi_nodei in enumerate(dNdxi):
                for jnode, dNdxi_nodej in enumerate(dNdxi):
                    ind_i, ind_j = inode*ndim, jnode*ndim
                    K_e[iel, ind_i:ind_i+ndim, ind_j:ind_j+ndim] += w[ixi]*J*model.E*np.outer(dNdxi_nodei, dNdxi_nodej)  # (ndim x ndim)
            # K_e[iel, :,:] += w[ixi]*model.E*dNdxi.T@dNdxi

            # eps_ = 1/2*(displacements.T@dNdx + dNdx.T@displacements)  # This line projects displacement at the nodes to the strain at the integration point
            # dsde = mat_stiffness(eps_, nnodes_el, ndim)

            # for A in range(nnodes_el):
            #     for i in range(ndim):
            #         for B in range(nnodes_el):
            #             for j in range(ndim):
            #                 K_e[ndim*A+i, ndim*B+j] += w[ixi] * J * dsde * (dNdx[A,i]*dNdx[B,j])  # (ndim*nnodes_el x ndim*nnodes_el)

    return K_e


def global_stiffness(model, K_e):

    nnodes, ndim = model.nnodes, model.ndim
    K_global = np.zeros((nnodes*ndim, nnodes*ndim))

    for iel, element in enumerate(model.elements):
        node_num = element
        # xs_node = model.nodes[node_num]                # (nnodes_el x ndim)
        for nodeii, nodei in enumerate(node_num):
            for nodejj, nodej in enumerate(node_num):
                ind_i, ind_j = nodei*ndim, nodej*ndim
                ind_ii, ind_jj = nodeii*ndim, nodejj*ndim
                K_global[ind_i:ind_i+ndim, ind_j:ind_j+ndim] = K_e[iel, ind_ii:ind_ii+ndim, ind_jj:ind_jj+ndim]

    return K_global


def global_force(model):




if __name__ == '__main__':

    model = QFEM('1D_static.inp')
    K_e= element_stiffness(model)
    global_stiffness(model, K_e)


    displacement = np.zeros(ndim*nnodes)
    K = global_stiffness(nodes, elements, displacement, nnodes, ndim)
    F = np.zeros((ndim*nnodes))
    ind = []
    for BC in BCs:
        ind.append(int(ndim*BC[0]+BC[1]))
    for i, val in enumerate(ind):
        F -= K[val] * BCs[i,2]
        F[val] = BCs[i,2]
        K[:,val], K[val,:] = 0, 0
        K[val,val] = 1
    # K = np.delete(K, ind, axis=0)
    # K = np.delete(K, ind, axis=1)

    print(K)
    U = np.linalg.inv(K)@F
    print(U)