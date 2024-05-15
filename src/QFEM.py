import numpy as np
from io import StringIO
import matplotlib.pyplot as plt

import utils


class QFEM():
    def __init__(self, inp_fpath, inp_format, el_type):

        inp_format = inp_format.lower()
        assert inp_format in ['bower', 'abq']

        self.nodes, self.elements, self.ndim, self.nnodes, self.nnodes_el, \
            self.nel, self.props, self.BCs, self.tractions = utils.read_inp(inp_fpath, inp_format)

        self.model_check(self, el_type)

    def model_check(self, el_type):

        if el_type=='CPS4':
            assert self.ndim==2
            assert self.nnodes_el==4
            assert self.props['PSPE']==0
        elif el_type=='CPE4':
            assert self.ndim==2
            assert self.nnodes_el==4
            assert self.props['PSPE']==1
        elif el_type=='C3D8':
            assert self.ndim==3
            assert self.nnodes_el==8
        elif el_type=='C3D20':
            assert self.ndim==3
            assert self.nnodes_el==20

        return True









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


def element_matrices(model):

    nnodes_el, ndim = model.nnodes_el, model.ndim
    xi_list, w = integration_points_weights(nnodes_el, ndim, reduced=False)
    K_e = np.zeros((model.nel, nnodes_el*ndim, nnodes_el*ndim))
    f_e = np.zeros((model.nel, nnodes_el*ndim))

    for iel, element in enumerate(model.elements):
        node_num = element
        xs_node = model.nodes[node_num]                # (nnodes_el x ndim)
        for ixi, xi in enumerate(xi_list):
            # This part are performed at the integration point
            # xs_int  = xs_node.T@shape_func(xi, nnodes_el, ndim)    # (ndim x 1) integration point coords
            N = shape_func(xi, nnodes_el, ndim)            # (nnodes_el x ndim)
            dNdxi = shape_func_deriv(xi, nnodes_el, ndim)            # (nnodes_el x ndim)
            dxdxi = xs_node.T@dNdxi               # (ndim x ndim)
            # dxidx = np.linalg.inv(dxdxi)          # (ndim x ndim)
            # dNdx  = dNdxi@dxidx                   # (nnodes_el x ndim)
            J = np.linalg.det(dxdxi)
            for inode, dNdxi_inode in enumerate(dNdxi):  # inode is the i-th node in the element
                ind_i = inode*ndim

                # Element force vector
                bf = model.body_force[iel]  # (ndim x 1)
                f_e[iel, ind_i:ind_i+ndim] += bf * w[ixi]*N[inode]*J  # (ndim x 1)

                for jnode, dNdxi_jnode in enumerate(dNdxi):
                    ind_j = jnode*ndim

                    # Element stiffness matrix
                    K_e[iel, ind_i:ind_i+ndim, ind_j:ind_j+ndim] += w[ixi]/J*model.E*np.outer(dNdxi_inode, dNdxi_jnode)  # (ndim x ndim)
            # K_e[iel, :,:] += w[ixi]*model.E*dNdxi.T@dNdxi

            # eps_ = 1/2*(displacements.T@dNdx + dNdx.T@displacements)  # This line projects displacement at the nodes to the strain at the integration point
            # dsde = mat_stiffness(eps_, nnodes_el, ndim)

            # for A in range(nnodes_el):
            #     for i in range(ndim):
            #         for B in range(nnodes_el):
            #             for j in range(ndim):
            #                 K_e[ndim*A+i, ndim*B+j] += w[ixi] * J * dsde * (dNdx[A,i]*dNdx[B,j])  # (ndim*nnodes_el x ndim*nnodes_el)

    return K_e, f_e


def global_matrices(model, K_e, f_e):

    nnodes, ndim = model.nnodes, model.ndim
    K_global = np.zeros((nnodes*ndim, nnodes*ndim))
    f_global = np.zeros((nnodes*ndim))

    for iel, element in enumerate(model.elements):
        node_num = element
        # xs_node = model.nodes[node_num]                # (nnodes_el x ndim)
        for inode, nodei in enumerate(node_num):  # nodei is the node number, inode is the i-th node in element definition
            ind_ii, ind_i = nodei*ndim, inode*ndim
            f_global[ind_ii:ind_ii+ndim] += f_e[iel, ind_i:ind_i+ndim]

            for jnode, nodej in enumerate(node_num):
                ind_jj, ind_j = nodej*ndim, jnode*ndim
                K_global[ind_ii:ind_ii+ndim, ind_jj:ind_jj+ndim] += K_e[iel, ind_i:ind_i+ndim, ind_j:ind_j+ndim]

    return K_global, f_global


def apply_boundary_conditions(model, K_global, f_global):

    # Update the global force vector with the traction term
    K_mod, f_mod = K_global, f_global
    for traction in model.traction:
        node = int(traction[0])
        f_mod[node:node+model.ndim] += traction[1:]

    # Displacement boundary conditions
    for BC in model.BC:
        row = int(BC[0]*model.ndim + BC[1])
        K_mod[row,:] = 0
        K_mod[row,row] = 1
        f_mod[row] = BC[2]

    return K_mod, f_mod


if __name__ == '__main__':

    model = QFEM('1D_static.inp')
    K_e, f_e = element_matrices(model)
    K_global, f_global = global_matrices(model, K_e, f_e)
    K_mod, f_mod = apply_boundary_conditions(model, K_global, f_global)

    # displacement = np.zeros(ndim*nnodes)
    # K = global_stiffness(nodes, elements, displacement, nnodes, ndim)
    # F = np.zeros((ndim*nnodes))
    # ind = []
    # for BC in BCs:
    #     ind.append(int(ndim*BC[0]+BC[1]))
    # for i, val in enumerate(ind):
    #     F -= K[val] * BCs[i,2]
    #     F[val] = BCs[i,2]
    #     K[:,val], K[val,:] = 0, 0
    #     K[val,val] = 1
    # K = np.delete(K, ind, axis=0)
    # K = np.delete(K, ind, axis=1)

    U = np.linalg.inv(K_mod)@f_mod
    print(U)
    np.savetxt("K_mod.csv", K_mod, delimiter=",")
    np.savetxt("f_mod.csv", f_mod, delimiter=",")

    fig, ax = plt.subplots()
    ax.plot(model.nodes, U, marker='o')
    ax.set_xlabel('x')
    ax.set_ylabel('Displacement')

    fig.show()

