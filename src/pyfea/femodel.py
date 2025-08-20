import os
import numpy as np
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt

from pyfea import utils
from pyfea.material import Material
from pyfea.elements.Elements import Element

OUT_PATH = "output"

class FEModel():
    def __init__(self, inp_fpath: os.PathLike, inp_format: str, el_type: str | None = None, job_name: str | None = None):

        inp_format = inp_format.lower()
        assert inp_format in ['bower', 'abq']

        self.nodes, self.elements, self.ndim, self.nnodes, self.nelnodes, \
            self.nel, self.props, self.nodal_disp, self.tractions, self.body_force = utils.read_inp(inp_fpath, inp_format)

        # list of [Node_#, DOF#, Value]
        # list of [Element_#, Face_#, Traction_components]
        # list of [Element_#, force vector] force vector

        self.reduced_integration = False
        self.job_name = job_name or os.path.basename(inp_fpath).split('.')[0]

        if el_type is not None:
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

    def set_up_materials(self):
        self.material = Material(self.props, ndim=self.ndim)

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
            for inode, nodei in enumerate(element.nodes):
                for jnode, nodej in enumerate(element.nodes):
                    K[nodei*self.ndim:(nodei+1)*self.ndim, nodej*self.ndim:(nodej+1)*self.ndim] += \
                        K_e[inode*self.ndim:(inode+1)*self.ndim, jnode*self.ndim:(jnode+1)*self.ndim]

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
        self.post_process()

    def set_solution(self):
        for element in self.Elements.values():
            element.set_solution(self.U[element.nodes])

    def reaction_forces(self):
        rf = []
        for node, dof, disp in self.nodal_disp:
            row = int(node*self.ndim + dof)
            rf.append(self.K[row, :] @ self.U.flatten() - self.f[row])
        rf = np.array(rf)
        self.rf = pd.DataFrame(dict(
            node = self.nodal_disp[:, 0],
            dof = self.nodal_disp[:, 1],
            disp = self.nodal_disp[:, 2],
            reaction_force = rf
        ))
        return self.rf

    def post_process(self):
        RF = self.reaction_forces()
        for iel, element in self.Elements.items():
            element.post_process()

    def write_results(self, out_name: str | None = None):
        out_name = out_name or f"{self.job_name}_results.csv"
        df = pd.DataFrame(columns=["element", "integration_point", "x", "y", "e_11", "e_22", "e_12", "s_11", "s_22", "s_12"])
        for iel, element in self.Elements.items():
            for ixi, xi in enumerate(element.xi_int):
                d = {
                    "element": iel,
                    "integration_point": ixi,
                    "x": element.x_int[ixi][0],
                    "y": element.x_int[ixi][1],
                }
                for i in range(self.ndim):
                    for j in range(i, self.ndim):
                        d[f"e_{i+1}{j+1}"] = element.strain[ixi][i, j]
                        d[f"s_{i+1}{j+1}"] = element.stress[ixi][i, j]
                df = pd.concat([df, pd.DataFrame(d, index=[0])], ignore_index=True)
        df.to_csv(os.path.join(OUT_PATH, out_name), index=False, float_format='%.4f')

    def plot_domain(self, out_name: str | None = None, out_path: os.PathLike | None = None):
        out_name = out_name or f"{self.job_name}_domain.png"
        out_path = out_path or OUT_PATH

        fig, ax = plt.subplots()

        utils.plot_mesh(self.Elements, ax=ax, config="Initial", color='green', marker='o', linestyle="-")
        utils.plot_mesh(self.Elements, ax=ax, config="Deformed", color='red', marker='x', linestyle="-")

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(self.job_name)

        h, l = ax.get_legend_handles_labels()
        h = [h[0], h[len(self.Elements)]]  # Keep only one legend for each configuration
        l = [l[0], l[len(self.Elements)]]  # Keep only one legend for each configuration
        ax.legend(h[:2], l[:2], frameon=False)
        plt.savefig(os.path.join(out_path, out_name))
        # plt.close()
        return fig, ax

if __name__ == '__main__':

    # inp_fpath = './MatLab sample code_Bower/Linear_elastic_quad4.txt'
    inp_fpath = 'src/sample_code_matlab_Bower/linear_elastic_Brick8.txt'
    inp_fpath = 'src/sample_code_matlab_Bower/shear_locking_demo.txt'
    model = FEModel(inp_fpath=inp_fpath, inp_format='bower')  # inp_fpath, inp_format, el_type
    model.set_up()
    model.solve()
    model.write_results()
    model.plot_domain()
