import os
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt

import utils
from material import Material
from elements.Elements import Element

OUT_PATH = "output"

class FEModel():
    def __init__(self, inp_fpath: os.PathLike, inp_format: str, el_type: str, job_name: str | None = None):

        inp_format = inp_format.lower()
        assert inp_format in ['bower', 'abq']

        self.nodes, self.elements, self.ndim, self.nnodes, self.nelnodes, \
            self.nel, self.props, self.nodal_disp, self.tractions, self.body_force = utils.read_inp(inp_fpath, inp_format)

        # list of [Node_#, DOF#, Value]
        # list of [Element_#, Face_#, Traction_components]
        # list of [Element_#, force vector] force vector

        self.reduced_integration = False
        self.model_check(el_type)
        self.job_name = job_name or os.path.basename(inp_fpath).split('.')[0]

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

    def set_up_materials(self):
        self.material = Material(E=self.props['E'], nu=self.props['nu'], ndim=self.ndim, pspe=self.props['PSPE'])

    def set_up_elements(self):
        self.Elements = {}
        for iel, el in enumerate(self.elements):
            element = Element(label=iel, nodes=el, nodes_x=self.nodes[el], material=self.material, reduced_integration=self.reduced_integration)
            self.Elements[iel] = element

    def set_up_force_BCs(self):
        if self.tractions is not None:
            for i, iel in enumerate(self.tractions[:, 0]):
                self.Elements[iel].set_tractions(self.tractions[i])

        if self.body_force is not None:
            for i, iel in enumerate(self.body_force[:, 0]):
                self.Elements[iel].set_body_force(self.body_force[i])

    def global_stiffness_matrix(self):
        K = np.zeros((self.ndim*self.nnodes, self.ndim*self.nnodes))
        for iel, element in self.Elements.items():
            K_e = element.element_stiffness_matrix()
            for inode, node in enumerate(element.nodes):
                K[node*self.ndim:(node+1)*self.ndim, node*self.ndim:(node+1)*self.ndim] += \
                    K_e[inode*self.ndim:(inode+1)*self.ndim, inode*self.ndim:(inode+1)*self.ndim]

        self.K = K
        return K

    def global_force_vector(self):
        f = np.zeros((self.ndim*self.nnodes))
        for iel, element in self.Elements.items():
            f_e = element.element_force_vector()
            for inode, node in enumerate(element.nodes):
                f[node*self.ndim:(node+1)*self.ndim] += f_e[inode*self.ndim:(inode+1)*self.ndim]

        self.f = f
        return f

    def set_up(self):
        self.set_up_materials()
        self.set_up_elements()
        self.set_up_force_BCs()

    def solve(self):

        K = self.global_stiffness_matrix()
        f = self.global_force_vector()

        # [Node_#, DOF#, Value]
        # Apply displacement boundary conditions
        for node, dof, disp in self.nodal_disp:
            row = int(node*self.ndim + dof)
            assert f[row] == 0, f"Force at node {node}, dof {dof} is not zero before applying boundary condition."
            K[row, :] = 0  # Zero out the row
            # K[:, loc] = 0  # Zero out the column
            K[row, row] = 1  # Set the diagonal to 1
            f[row] = disp  # Enforce the displacement

        U = np.linalg.solve(K, f)
        self.U = U.reshape((self.nnodes, self.ndim))
        self.set_solution()
        return self.U

    def set_solution(self):
        for element in self.Elements.values():
            element.set_solution(self.U[element.nodes])

    def reaction_forces(self):
        rf = []
        for node, dof, disp in self.nodal_disp:
            row = node*self.ndim + dof
            rf.append(self.K[row, :] @ self.U.flatten() - self.f[row])
        self.rf = np.array(rf)
        return self.rf

    def post_analysis(self):
        RF = self.reaction_forces()
        for iel, element in self.Elements.items():
            element.get_strain()

    def plot_domain(self, out_name: str | None = None):
        out_name = out_name or f"{self.job_name}_domain.png"
        fig, ax = plt.subplots()
        for iel, element in self.Elements.items():
            nodes_x = element.nodes_x
            nodes_x0 = element.nodes_x0
            ax.plot(nodes_x[:, 0], nodes_x[:, 1], 'go-', label=f'Element {iel}')
            ax.plot(nodes_x0[:, 0], nodes_x0[:, 1], 'rx--', label=f'Element {iel} (Initial)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Deformed and Initial Configuration')
        # ax.legend()
        plt.savefig(os.path.join(OUT_PATH, out_name))
        plt.close()

if __name__ == '__main__':

    # inp_fpath = './MatLab sample code_Bower/Linear_elastic_quad4.txt'
    inp_fpath = 'src/sample_code_matlab_Bower/Linear_elastic_quad4.txt'
    model = FEModel(inp_fpath=inp_fpath, inp_format='bower', el_type='CPE4')  # inp_fpath, inp_format, el_type
    model.set_up()
    model.solve()
    model.plot_domain()

    # K_e, f_e = utils.element_matrices(model)
    # K_global, f_global = utils.global_matrices(model, K_e, f_e)
    # K_mod, f_mod = utils.apply_boundary_conditions(model, K_global, f_global)

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

    # U = np.linalg.inv(K_mod)@f_mod
    # print(U)
    # np.savetxt("K_mod.csv", K_mod, delimiter=",")
    # np.savetxt("f_mod.csv", f_mod, delimiter=",")

    # fig, ax = plt.subplots()
    # ax.plot(model.nodes, U, marker='o')
    # ax.set_xlabel('x')
    # ax.set_ylabel('Displacement')

    # fig.show()

