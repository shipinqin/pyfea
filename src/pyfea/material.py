import numpy as np

class Material:

    def __init__(self, props: dict, ndim: int):

        self.E = props.get('E', None)  # Young's modulus
        self.nu = props.get('nu', None)  # Poisson's ratio
        self.mu = props.get('mu', None) or self.E/(2*(1+self.nu))  # shear modulus
        self.ndim = ndim
        self.pspe = int(props.get('PSPE', 0))  # follow Bower convention, 0 means PS, 1 means PE

        self.C = self.stiffness()

    def stiffness(self) -> np.ndarray:

        if self.ndim ==3:
            C = np.zeros((3, 3, 3, 3))
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        for l in range(3):
                            if i==k and j==l:
                                C[i,j,k,l] = C[i,j,k,l] + self.mu
                            if i==j and k==l:
                                C[i,j,k,l] = C[i,j,k,l] + 2*self.mu*self.nu/(1-2*self.nu)
                            if i==l and j==k:
                                C[i,j,k,l] = C[i,j,k,l] + self.mu
        elif self.ndim == 2:
            C = np.zeros((2, 2, 2, 2))
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        for l in range(2):
                            if i==k and j==l:
                                C[i,j,k,l] = C[i,j,k,l] + self.mu
                            if i==j and k==l:
                                if self.pspe == 1:  # plane strain
                                    C[i,j,k,l] = C[i,j,k,l] + 2*self.mu*self.nu/(1-2*self.nu)
                                else:  # plane stress
                                    C[i,j,k,l] = C[i,j,k,l] + 2*self.mu*self.nu/(1-self.nu)
                            if i==l and j==k:
                                C[i,j,k,l] = C[i,j,k,l] + self.mu
        elif self.ndim == 1:
            C = np.array([self.E])

        return C

    def get_stiffness(self, strain: np.ndarray, dstrain: np.ndarray) -> np.ndarray:
        dsde = self.C
        return dsde

    def get_stress(self, strain: np.ndarray, dstrain: np.ndarray) -> np.ndarray:
        dsde = self.get_stiffness(strain, dstrain)
        return np.einsum('ijkl,kl->ij', dsde, (strain + dstrain))