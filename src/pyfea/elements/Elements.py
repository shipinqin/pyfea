import numpy as np
from typing import Sequence

from elements import shape_func
from material import Material
import utils

class Element:

    def __init__(self, label: int, nodes: np.ndarray, nodes_x: np.ndarray, material: Material, reduced_integration: bool = False):

        self.label = label
        self.nodes = nodes  # shape (nelnodes, )
        self.nodes_x0 = nodes_x  # shape (nelnodes, ndim)
        self.nodes_x = nodes_x  # shape (nelnodes, ndim)
        self.nodes_u = np.zeros_like(nodes_x)  # Displacement at current step, shape (nelnodes, ndim)
        self.nodes_du = np.zeros_like(nodes_x)  # Displacement inc at current step, shape (nelnodes, ndim)
        self.nelnodes, self.ndim = nodes_x.shape
        self.reduced_integration = reduced_integration

        self.material = material

        self.N = shape_func.get_shape_func(self.nelnodes, self.ndim)  # shape (nelnodes, )
        self.dNdxi = shape_func.get_shape_func_deriv(self.nelnodes, self.ndim) # shape (nelnodes, ndim)
        self.xi_int, self.w_int = shape_func.get_integration_points_weights(self.nelnodes, self.ndim, reduced_integration)
        self.x_int = self.get_x_at_int()  # shape (nint, ndim)

        self._Jacobian = None  # This saves the Jacobian at integration points, in the same order as xi_int
        self._Jacobian_inv = None  # This saves the inverse of Jacobian at integration points, in the same order as xi_int
        self._Jacobian_det = None  # This saves the determinant of Jacobian at integration points, in the same order as xi_int

        self.nodal_disp = None
        self.tractions = None
        self.body_force = 0.0

        self.strain = np.zeros((len(self.xi_int), self.ndim, self.ndim))
        self.stress = np.zeros((len(self.xi_int), self.ndim, self.ndim))
        self.dstrain = np.zeros((len(self.xi_int), self.ndim, self.ndim))

    def set_solution(self, U):
        self.nodes_du = U  # This is the displacement in the current inc

    def post_process(self):
        for ixi, xi in enumerate(self.xi_int):
            self.dstrain[ixi] = self.get_dstrain(xi)
            self.stress[ixi] = self.get_stress(xi)
        self.strain += self.dstrain  # Total strain

        # Update status
        self.nodes_u += self.nodes_du  # Total displacement
        self.nodes_x = self.nodes_x + self.nodes_du  # Current position

    def set_nodal_disp(self, nodal_disp: np.ndarray):
        self.nodal_disp = nodal_disp

    def set_tractions(self, tractions: np.ndarray):
        # tractions: [Element_#, Face_#, Traction_components]
        self.traction_nodes = utils.get_face_nodes(self.ndim, self.nelnodes, tractions[1])  # This stores the face nodes in terms of their sequence in the element definition
        self.tractions = tractions[2:]  # shape (ndim, )

    def set_body_force(self, body_force: np.ndarray):
        self.body_force = body_force  # shape (ndim, )

    def update_nodes_x(self):
        self.nodes_x += self.nodes_du
        self.nodes_u += self.nodes_du

    def get_x(self, xi: np.ndarray) -> np.ndarray[np.float64]:
        assert len(xi) == self.ndim
        return self.N(xi)@self.nodes_x  # shape (ndim, )

    def get_Jacobian(self, xi: np.ndarray) -> np.ndarray[np.float64]:
        assert len(xi) == self.ndim
        return self.dNdxi(xi).T @ self.nodes_x  # shape (ndim, ndim)

    def get_dNdx(self, xi: np.ndarray) -> np.ndarray[np.float64]:
        assert len(xi) == self.ndim
        return self.dNdxi(xi) @ np.linalg.inv(self.get_Jacobian(xi))  # shape (nelnodes, ndim)

    def get_strain(self, xi: np.ndarray) -> np.ndarray[np.float64]:
        assert len(xi) == self.ndim
        return self.get_dNdx(xi).T @ self.nodes_u  # shape (ndim, ndim)

    def get_dstrain(self, xi: np.ndarray) -> np.ndarray[np.float64]:
        assert len(xi) == self.ndim
        return self.get_dNdx(xi).T @ self.nodes_du  # shape (ndim, ndim)

    def get_stress(self, xi: np.ndarray) -> np.ndarray[np.float64]:
        assert len(xi) == self.ndim
        strain = self.get_strain(xi)
        dstrain = self.get_dstrain(xi)
        return self.material.get_stress(strain, dstrain)  # shape (ndim, ndim)

    def get_consistent_stiffness(self, xi: np.ndarray) -> np.ndarray[np.float64]:
        assert len(xi) == self.ndim
        strain = self.get_strain(xi)
        dstrain = self.get_dstrain(xi)
        return self.material.get_stiffness(strain, dstrain)

    def get_x_at_int(self) -> np.ndarray[np.ndarray[np.float64]]:
        x_int = []
        for xi in self.xi_int:
            x_int.append(self.get_x(xi))
        return np.array(x_int)  # shape (nint, ndim)

    def get_N_at_int(self) -> np.ndarray[np.ndarray[np.float64]]:
        N_int = []
        for xi in self.xi_int:
            N_int.append(self.N(xi))
        return np.array(N_int)  # shape (nint, nelnodes)

    def get_dNdx_at_int(self) -> np.ndarray[np.ndarray[np.float64]]:
        dNdx_int = []
        for xi in self.xi_int:
            dNdx_int.append(self.get_dNdx(xi))
        return np.array(dNdx_int)  # shape (nint, nelnodes, ndim)

    def get_consistent_stiffness_at_int(self) -> np.ndarray[np.ndarray[np.float64]]:
        C_int = []
        for xi in self.xi_int:
            C_int.append(self.get_consistent_stiffness(xi))
        return np.array(C_int)  # shape (nint, ndim, ndim, ndim, ndim)

    def get_Jacobian_at_int(self) -> np.ndarray[np.ndarray[np.float64]]:
        if self._Jacobian is None:
            jacobian, jacobian_inv, jacobian_det = [], [], []
            for xi in self.xi_int:
                j = self.dNdxi(xi).T @ self.nodes_x
                jacobian.append(j)
                jacobian_inv.append(np.linalg.inv(j))
                jacobian_det.append(np.linalg.det(j))
            self._Jacobian = np.array(jacobian)
            self._Jacobian_inv = np.array(jacobian_inv)
            self._Jacobian_det = np.array(jacobian_det)

    def element_stiffness_matrix(self):

        K_e = np.zeros((self.nelnodes*self.ndim, self.nelnodes*self.ndim))

        dNdx_at_int = self.get_dNdx_at_int()  # shape (nint, nelnodes, ndim)
        C_at_int = self.get_consistent_stiffness_at_int()  # shape (nint, ndim, ndim, ndim, ndim)
        self.get_Jacobian_at_int()  # This will populate self._Jacobian, self._Jacobian_inv, self._Jacobian_det

        temp = np.zeros((self.nelnodes, self.ndim, self.ndim, self.nelnodes))  # shape (nelnodes, ndim, ndim, nelnodes)
        for ixi, xi in enumerate(self.xi_int):
            temp += self.w_int[ixi] * self._Jacobian_det[ixi] * \
                     np.einsum("ai,ijkl,lb->ajkb", dNdx_at_int[ixi, :, :], C_at_int[ixi], dNdx_at_int[ixi, :, :].T)  # shape (nelnodes, ndim, ndim, nelnodes)
        K_e = np.zeros((self.nelnodes*self.ndim, self.nelnodes*self.ndim))
        for inode in range(self.nelnodes):
            for jnode in range(self.nelnodes):
                K_e[inode*self.ndim:(inode+1)*self.ndim, jnode*self.ndim:(jnode+1)*self.ndim] = temp[inode, :, :, jnode]  # shape (ndim, ndim)

        self.K = K_e

        return self.K  # shape (nelnodes*ndim, nelnodes*ndim)

    def element_force_vector(self):

        f_e = np.zeros((self.nelnodes, self.ndim))

        if self.body_force is not None:
            bf = np.sum(self.get_N_at_int() * self.w_int[:, np.newaxis] * self._Jacobian_det[:, np.newaxis], axis=0)  # shape (nelnodes, )
            bf = np.outer(bf, self.body_force)  # shape (nelnodes, ndim)

        if self.tractions is not None:
            n_facenodes = len(self.traction_nodes)
            facenodes_x = self.nodes_x[self.traction_nodes]  # shape (n_facenodes, ndim)
            N_face = shape_func.get_shape_func(n_facenodes, self.ndim-1)  # shape (n_facenodes, )
            dNdxi_face = shape_func.get_shape_func_deriv(n_facenodes, self.ndim-1) # shape (n_facenodes, ndim-1)

            traction_e = np.zeros((n_facenodes, self.ndim))
            xi_int_face, w_int_face = shape_func.get_integration_points_weights(n_facenodes, self.ndim-1, self.reduced_integration)
            for ixi, xi in enumerate(xi_int_face):
                J_face = dNdxi_face(xi).T @ facenodes_x  # shape (ndim-1, ndim)
                J_face_det = utils.face_det(J_face, self.ndim)  # shape (1, )
                traction_e += w_int_face[ixi] * np.outer(N_face(xi), self.tractions) * J_face_det  # shape (n_facenodes, ndim)
            f_e[self.traction_nodes] += traction_e  # shape (nelnodes, ndim)

        self.f = f_e.flatten()  # shape (nelnodes*ndim, )
        return self.f  # shape (nelnodes*ndim, )
