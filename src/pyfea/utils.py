import os
import numpy as np

from femodel import FEModel


def read_inp(inp_fpath: os.PathLike, inp_format: str):

    assert inp_format in ['bower', 'abq']

    model = {}
    # If the input file is ABAQUS inp style
    if inp_format == 'abq':
        nodes, elements, ndim, nnodes, nelnodes, nel, \
            props, nodal_disp, tractions, body_force = read_abq_inp(inp_fpath)

        nint =2
        nshape = 3

        # Disp boundary conditions
        if nodal_disp is None:
            BC_reg = [0]
            BC_dof = [0]
            BC_mag = [0]
            nodal_disp = np.array([BC_reg, BC_dof, BC_mag]).T

        # Traction
        if tractions is None:
            traction_reg = [len(nodes)-1]
            traction_vec = [2]  # (ndim x 1)
            tractions = np.zeros((len(traction_reg), 1+ndim))
            tractions[:,0] = traction_reg
            tractions[:,1:] = traction_vec

        # Body force
        if body_force is None:
            body_force_reg = np.array(range(len(elements)))
            body_force_vec = [10]  #  (ndim x 1)

            body_force = np.zeros((nel, ndim))
            body_force[body_force_reg] = body_force_vec
            # body_force = np.array([[el, body_force_vec] for el in elements])
            # body_force = [nodes, 'x', 10]

        A = 1
        mu = 50
        nu = 0.3
        const = 2*mu*A*(1-nu)/(1-2*nu)
        E = const
        props = {'E': E, 'mu': mu, 'nu': nu, 'PSPE': 0}  # follow Bower convention, 0 means PS, 1 means PE

    elif inp_format == 'bower':
        nodes, elements, ndim, nnodes, nelnodes, nel, \
            props, nodal_disp, tractions, body_force = read_bower_inp(inp_fpath)
        if len(props) == 3:
            props = {'mu': props[0],
                     'nu': props[1],
                     'PSPE': props[2]}  # according to Bower, 0 means PS, 1 means PE
            props.update({'E': 2*props['mu']*(1+props['nu'])})

    return nodes, elements, ndim, nnodes, nelnodes, nel, props, nodal_disp, tractions, body_force


def get_value(line, type_):

    if type_=='float':
        return float(line.split(':')[1].strip())
    elif type_=='int':
        return int(line.split(':')[1].strip())
    else:
        raise Exception('Wrong type to extract')

def dict2array(d):
    return np.array([d[key] for key in sorted(d.keys())])

def read_abq_inp(inp_fpath: os.PathLike):
    # ChatGPT generated, Shipin revised

    if not inp_fpath.endswith('.inp'):
        inp_fpath.append('.inp')

    nodes, elements = [], []
    with open(inp_fpath, 'r') as file:
        lines = file.readlines()

        # Flag to indicate when to start reading nodes and elements
        read_nodes, read_elements = False, False

        for line in lines:
            # Start reading nodes
            if '*Node' in line:
                read_nodes = True
                read_elements = False
                continue
            # Start reading elements
            elif '*Element' in line:
                read_nodes = False
                read_elements = True
                continue

            # Read nodes
            if read_nodes:
                if line.strip():
                    node_info = line.split(',')
                    node_id = int(node_info[0])
                    if node_id == len(nodes)+1:
                        nodes.append(list(map(float, node_info[1:])))
                    else:
                        raise ValueError(f"Node IDs are not sequential: expected {len(nodes)+1}, got {node_id}")
                    # node_coords = list(map(float, node_info[1:]))
                    # nodes[int(node_id)] = node_coords

            # Read elements
            if read_elements:
                if line.strip():
                    element_info = line.split(',')
                    element_id = int(element_info[0])
                    if element_id == len(elements)+1:
                        elements.append(list(map(int, element_info[1:])))
                    else:
                        raise ValueError(f"Element IDs are not sequential: expected {len(elements)+1}, got {element_id}")
                    # elements[element_id] = node_ids

        nodes = np.array(nodes)
        nnodes, ndim = nodes.shape
        elements = np.array(elements) - 1  # Convert to 0-indexed
        nel, nelnodes = elements.shape
        # ndim, nnodes = len(nodes.values[0]), len(nodes)
        # nelnodes, nel = len(elements.values[0]), len(elements)

        nodal_disp = None
        tractions = None
        body_force = None

        props = None

    return nodes, elements, ndim, nnodes, nelnodes, nel, props, nodal_disp, tractions, body_force


def read_bower_inp(inp_fpath: os.PathLike):

    if not inp_fpath.endswith('.txt'):
        inp_fpath.append('.txt')

    nodes, elements = {}, {}
    with open(inp_fpath, 'r') as file:
        lines = file.readlines()

        i = 0

        # Material properties
        nprops = get_value(lines[i], 'int')
        props = np.zeros(nprops)
        i += 1

        for j in range(nprops):
            props[j] = get_value(lines[i+j], 'float')
        i += nprops

        # Basic model info
        ndim = get_value(lines[i], 'int')
        ndof = get_value(lines[i+1], 'int')
        nnodes = get_value(lines[i+2], 'int')
        i += 4

        # Node definition
        nodes = lines[i:i+nnodes]
        nodes = np.array([node.strip().split() for node in nodes]).astype('float')
        if len(nodes[0]) != ndim:
            raise Exception('Wrong input in node definition')
        i += nnodes

        # element info
        nel = get_value(lines[i], 'int')
        nelnodes = get_value(lines[i+1], 'int')
        i += 3

        # Element definition
        elements = lines[i:i+nel]
        elements = np.array([element.strip().split() for element in elements]).astype('int')
        el_identifier = elements[:,0]
        nelnodes_all = elements[:,1]
        elements = elements[:,2:] -1  # -1 because Python is 0-indexed
        if len(np.unique(nelnodes_all)) != 1 or nelnodes_all[0] != nelnodes:
            raise Exception('Wrong number of nodes in element definition')
        i += nel

        # BC
        nBCs = get_value(lines[i], 'int')
        i += 2

        # list of [Node_#, DOF#, Value]
        BCs = lines[i:i+nBCs]
        nodal_disp = np.array([BC.strip().split() for BC in BCs]).astype('float')
        nodal_disp[:,:2] -= 1  # Python is 0-indexed
        i += nBCs

        # Traction
        ntraction = get_value(lines[i], 'int')
        i += 2

        # list of [Element_#, Face_#, Traction_components]
        tractions = lines[i:i+ntraction]
        tractions = np.array([traction.strip().split() for traction in tractions]).astype('float')
        tractions[:,:1] -= 1  # Python is 0-indexed

        # list of [Element_#, force vector] force vector
        body_force = None

    return nodes, elements, ndim, nnodes, nelnodes, nel, props, nodal_disp, tractions, body_force


def nfacenodes(ndim, nelnodes):
#====================== No. nodes on element faces ================
#
#    Adapted from Bower's MatLab code
#    This procedure returns the number of nodes on each element face
#    for various element types.  This info is needed for computing
#    the surface integrals associated with the element traction vector
#
    if ndim == 2:
        if nelnodes == 3 or nelnodes == 4:
            n = 2
        elif (nelnodes == 6 or nelnodes == 8):
            n = 3

    elif (ndim == 3):
        if (nelnodes == 4):
            n = 3
        elif (nelnodes == 10):
            n = 6
        elif (nelnodes == 8):
            n = 4
        elif (nelnodes == 20):
            n = 8

    return n

def get_face_nodes(ndim, nelnodes, face):
#======================= Lists of nodes on element faces =============
#
#    Adapted from Bower's MatLab code
#    This procedure returns the list of nodes on an element face
#    The nodes are ordered so that the element face forms either
#    a 1D line element or a 2D surface element for 2D or 3D problems
#
    # Added a fictitious -1 to i3 and i4 since python is 0-indexed
    i3 = [-1, 2,3,1]
    i4 = [-1, 2,3,4,1]
    face = int(face)

    face_nodes = np.zeros((nfacenodes(ndim,nelnodes)))

    if (ndim == 2):
        if (nelnodes == 3):
            face_nodes[0] = face
            face_nodes[1] = i3[face]
        elif (nelnodes == 6):
            face_nodes[0] = face
            face_nodes[1] = i3[face]
            face_nodes[2] = face+3
        elif (nelnodes==4):
            face_nodes[0] = face
            face_nodes[1] = i4[face]
        elif (nelnodes==8):
            face_nodes[0] = face
            face_nodes[1] = i4[face]
            face_nodes[2] = face+4

    elif (ndim == 3):
        if (nelnodes==4):
            if   (face == 1):
                face_nodes = [1,2,3]
            elif (face == 2):
                face_nodes = [1,4,2]
            elif (face == 3):
                face_nodes = [2,4,3]
            elif (face == 4):
                face_nodes = [3,4,1]

        elif (nelnodes == 10):
            if   (face == 1):
                face_nodes = [1,2,3,5,6,7]
            elif (face == 2):
                face_nodes = [1,4,2,8,9,5]
            elif (face == 3):
                face_nodes = [2,4,3,9,10,6]
            elif (face == 4):
                face_nodes = [3,4,1,10,8,7]

        elif (nelnodes == 8):
            if   (face == 1):
                face_nodes = [1,2,3,4]
            elif (face == 2):
                face_nodes = [5,8,7,6]
            elif (face == 3):
                face_nodes = [1,5,6,2]
            elif (face == 4):
                face_nodes = [2,3,7,6]
            elif (face == 5):
                face_nodes = [3,7,8,4]
            elif (face == 6):
                face_nodes = [4,8,5,1]

        elif (nelnodes == 20):
            if   (face == 1):
                face_nodes = [1,2,3,4,9,10,11,12]
            elif (face == 2):
                face_nodes = [5,8,7,6,16,15,14,13]
            elif (face == 3):
                face_nodes = [1,5,6,2,17,13,18,9]
            elif (face == 4):
                face_nodes = [2,6,7,3,18,14,19,10]
            elif (face == 5):
                face_nodes = [3,7,8,4,19,15,20,11]
            elif (face == 6):
                face_nodes = [4,8,5,1,20,16,17,12]

    face_nodes = np.array(face_nodes) - 1  # Python is zero indexed.
    return face_nodes.astype('int')

def face_det(J_face: np.ndarray, ndim: int) -> float:
    # determinant for Jacobian on the surface, which projects the 3D element onto the 2D face
    # Calc following Bower matlab code

    if ndim == 2:  # J_face.shape = (1, 2)
        det = np.sqrt(np.sum(J_face**2))
    elif ndim == 3:  # J_face.shape = (2, 3)
        det = np.sqrt(np.linalg.det(J_face[:,:2])**2+
                      np.linalg.det(J_face[:,1:])**2+
                      np.linalg.det(J_face[:,0:3:2])**2)
    return det


if __name__ == '__main__':

    inp_fpath = './MatLab sample code_Bower/Linear_elastic_quad4.txt'
    read_inp(inp_fpath, 'bower')