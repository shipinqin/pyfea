import numpy as np
from io import StringIO
import matplotlib.pyplot as plt

import utils


class QFEM():
    def __init__(self, inp_fpath, inp_format, el_type):

        inp_format = inp_format.lower()
        assert inp_format in ['bower', 'abq']

        self.nodes, self.elements, self.ndim, self.nnodes, self.nelnodes, \
            self.nel, self.props, self.BCs, self.tractions, self.body_force = utils.read_inp(inp_fpath, inp_format)
        self.reduced_int = False

        self.model_check(el_type)

    def model_check(self, el_type):

        if el_type=='CPS4':
            assert self.ndim==2
            assert self.nelnodes==4
            assert self.props['PSPE']==0
        elif el_type=='CPE4':
            assert self.ndim==2
            assert self.nelnodes==4
            assert self.props['PSPE']==1
        elif el_type=='C3D8':
            assert self.ndim==3
            assert self.nelnodes==8
        elif el_type=='C3D20':
            assert self.ndim==3
            assert self.nelnodes==20

        return True


def mat_stiffness(eps_, nelnodes, ndim, E=201000, nu=0.3):

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


if __name__ == '__main__':

    inp_fpath = './MatLab sample code_Bower/Linear_elastic_quad4.txt'
    model = QFEM(inp_fpath=inp_fpath, inp_format='bower', el_type='CPE4')  # inp_fpath, inp_format, el_type
    K_e, f_e = utils.element_matrices(model)
    K_global, f_global = utils.global_matrices(model, K_e, f_e)
    K_mod, f_mod = utils.apply_boundary_conditions(model, K_global, f_global)

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

