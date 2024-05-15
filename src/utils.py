import numpy as np
from io import StringIO


def read_inp(inp_fpath, inp_format):

    assert inp_format in ['bower', 'abq']

    model = {}
    # If the input file is ABAQUS inp style
    if inp_format == 'abq':
        nodes, elements, ndim, nnodes, nnodes_el, nel = read_abq_inp(inp_fpath)

        nint =2
        nshape = 3

        # Disp boundary conditions
        BC_reg = [0]
        BC_dof = [0]
        BC_mag = [0]
        BC = np.array([BC_reg, BC_dof, BC_mag]).T

        # Traction
        traction_reg = [len(nodes)-1]
        traction_vec = [2]  # (ndim x 1)
        traction = np.zeros((len(traction_reg), 1+ndim))
        traction[:,0] = traction_reg
        traction[:,1:] = traction_vec

        # Body force
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

    elif inp_format == 'bower':
        nodes, elements, ndim, nnodes, nnodes_el, nel, props, BCs, tractions = read_bower_inp(inp_fpath)
        if len(props) == 3:
            props = {'G': props[0],
                     'mu': props[1],
                     'PSPE': props[2]}  # according to Bower, 0 means PS, 1 means PE

    return nodes, elements, ndim, nnodes, nnodes_el, nel, props, BCs, tractions


def get_value(line, type_):

    if type_=='float':
        return float(line.split(':')[1].strip())
    elif type_=='int':
        return int(line.split(':')[1].strip())
    else:
        raise Exception('Wrong type to extract')


def read_abq_inp(inp_fpath):
    # ChatGPT generated, Shipin revised

    if not inp_fpath.endswith('.inp'):
        inp_fpath.append('.inp')

    nodes, elements = {}, {}
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
                    node_coords = list(map(int, node_info[1:]))
                    nodes[int(node_id)] = node_coords

            # Read elements
            if read_elements:
                if line.strip():
                    element_info = line.split(',')
                    element_id = int(element_info[0])
                    node_ids = list(map(int, element_info[1:]))
                    elements[element_id] = node_ids

        ndim, nnodes = len(nodes.values[0]), len(nodes)
        nnodes_el, nel = len(elements.values[0]), len(elements)

    return nodes, elements, ndim, nnodes, nnodes_el, nel


def read_bower_inp(inp_fpath):

    if not inp_fpath.endswith('.txt'):
        inp_fpath.append('.txt')

    nodes, elements = {}, {}
    with open(inp_fpath, 'r') as file:
        lines = file.readlines()

        i = 0

        nprops = get_value(lines[i], 'int')
        props = np.zeros(nprops)
        i += 1

        for j in range(nprops):
            props[j] = get_value(lines[i+j], 'float')
        i += nprops

        ndim = get_value(lines[i], 'int')
        ndof = get_value(lines[i+1], 'int')
        nnodes = get_value(lines[i+2], 'int')
        i += 4

        nodes = lines[i:i+nnodes]
        nodes = np.array([node.strip().split() for node in nodes]).astype('float')
        if len(nodes[0]) != ndim:
            raise Exception('Wrong input in node definition')
        i += nnodes

        nel = get_value(lines[i], 'int')
        nnodes_el = get_value(lines[i+1], 'int')
        i += 3

        elements = lines[i:i+nel]
        elements = np.array([element.strip().split() for element in elements]).astype('int')
        el_identifier = elements[:,0]
        nnodes_els = elements[:,1]
        elements = elements[:,2:]
        if len(np.unique(nnodes_els)) != 1 or nnodes_els[0] != nnodes_el:
            raise Exception('Wrong number of nodes in element definition')
        i += nel

        nBCs = get_value(lines[i], 'int')
        i += 2

        BCs = lines[i:i+nBCs]
        BCs = np.array([BC.strip().split() for BC in BCs]).astype('float')
        i += nBCs

        ntraction = get_value(lines[i], 'int')
        i += 2

        tractions = lines[i:i+ntraction]
        tractions = np.array([traction.strip().split() for traction in tractions]).astype('float')

    return nodes, elements, ndim, nnodes, nnodes_el, nel, props, BCs, tractions


def shape_func(xi, nnodes_el, ndim):

    # Shape function sequence should be in accordance with the node sequence in element connectivity!

    # xi (1 x ndim) is the local coordinate
    N = np.zeros(nnodes_el)

    # ------------------------------
    # 1D elements
    if ndim == 1:
        if nnodes_el == 2:  # C1D2
            N = np.array([(1-xi)/2, (1+xi)/2])
        elif nnodes_el == 3:  # C1D3
            N = np.array([-xi*(1-xi)/2, xi*(1+xi)/2, (1-xi)*(1+xi)])

    # ------------------------------
    # 2D elements
    if ndim == 2:
        if nnodes_el == 3:  # 1st order triangle
            N = np.array([xi[0], xi[1], 1-xi[0]-xi[1]])
        elif nnodes_el == 6:  # 2nd order triangle
            xi2 = 1-xi[0]-xi[1]
            N[0] = (2*xi[0]-1)*xi[0]
            N[1] = (2*xi[1]-1)*xi[1]
            N[2] = (2*xi2-1)*xi2
            N[3] = 4*xi[0]*xi[1]
            N[4] = 4*xi[1]*xi2
            N[5] = 4*xi[0]*xi2
        elif nnodes_el == 4:  # 1st order quad
            N[0] = (1-xi[0])*(1-xi[1])/4
            N[1] = (1+xi[0])*(1-xi[1])/4
            N[2] = (1+xi[0])*(1+xi[1])/4
            N[3] = (1-xi[0])*(1+xi[1])/4
        elif nnodes_el == 8:  # 2nd order quad
            N[0] = -(1-xi[0])*(1-xi[1])*(1+xi[0]+xi[1])/4
            N[1] = (1+xi[0])*(1-xi[1])*(xi[0]-xi[1]-1)/4
            N[2] = (1+xi[0])*(1+xi[1])*(xi[0]+xi[1]-1)/4
            N[3] = (1-xi[0])*(1+xi[1])*(xi[0]-xi[1]-1)/4
            N[4] = (1-xi[0]*xi[0])*(1-xi[1])/2
            N[5] = (1+xi[0])*(1-xi[1]*xi[1])/2
            N[6] = (1-xi[0]*xi[0])*(1+xi[1])/2
            N[7] = (1-xi[0])*(1-xi[1]*xi[1])/2

    # ------------------------------
    # 3D elements
    if ndim == 3:
        if nnodes_el == 8:  # C3D8
            for n_local in range(1,9):
                T1 = 1-xi[0] if n_local in [1, 4, 5, 8] else 1+xi[0]
                T2 = 1-xi[1] if n_local in [1, 2, 5, 6] else 1+xi[1]
                T3 = 1-xi[2] if n_local in [1, 2, 3, 4] else 1+xi[2]
                N[n_local-1] = 1/8*T1*T2*T3

    else:
        raise f'Model does not support ndim={ndim} and nnodes_el={nnodes_el}'

    return N


def shape_func_deriv(xi, nnodes_el, ndim):

    # Shape function sequence should be in accordance with the node sequence in element connectivity!

    # x.T*shape_dev gives dx/dxi
    dNdxi = np.zeros((nnodes_el, ndim))

    # ------------------------------
    # 1D elements
    if ndim == 1:
        if nnodes_el == 2:  # C1D2
            dNdxi = np.array([[-1/2], [1/2]])
        elif nnodes_el == 3:  # C1D3
            dNdxi = np.array([xi-1/2, xi+1/2, -2*xi])

    # ------------------------------
    # 2D elements
    elif ndim == 2:
        if nnodes_el == 3:  # 1st order triangle
            dNdxi[0] = [1, 0]
            dNdxi[1] = [0, 1]
            dNdxi[2] = [-1, -1]
        elif nnodes_el == 6:  # 2nd order triangle
            xi2 = 1-xi[0]-xi[1]
            dNdxi[0] = [4*xi[0]-1, 0]
            dNdxi[1] = [0, 4*xi[1]-1]
            dNdxi[2] = [-(4*xi2-1), -(4*xi2-1)]
            dNdxi[3] = [4*xi[1], 4*xi[0]]
            dNdxi[4] = [-4*xi[1], 4*(xi2-xi[1])]
            dNdxi[5] = [4*(xi2-xi[0]), -4*xi[0]]
            # In Bower's code, dNdxi[4] and dNdxi[5] are wrong
        elif nnodes_el == 4:  # 1st order quad
            dNdxi[0] = [-(1-xi[1])/4, -(1-xi[0])/4]
            dNdxi[1] = [(1-xi[1])/4, -(1+xi[0])/4]
            dNdxi[2] = [(1+xi[1])/4, (1+xi[0])/4]
            dNdxi[3] = [-(1+xi[1])/4, (1-xi[0])/4]
        elif nnodes_el == 8:  # 2nd order quad
            dNdxi[0] = [(1-xi[1]*(2*xi[0]+xi[1]))/4, (1-xi[0]*(xi[0]+2*xi[1]))/4]
            dNdxi[1] = [(1-xi[1]*(2*xi[0]-xi[1]))/4, (1+xi[0]*(-xi[0]+2*xi[1]))/4]
            dNdxi[2] = [(1+xi[1]*(2*xi[0]+xi[1]))/4, (1+xi[0]*(xi[0]+2*xi[1]))/4]
            dNdxi[3] = [(1+xi[1]*(2*xi[0]-xi[1]))/4, (1-xi[0]*(-xi[0]+2*xi[1]))/4]
            dNdxi[4] = [-xi[0]*(1-xi[1]), -(1-xi[0]*xi[0])/2]
            dNdxi[5] = [(1-xi[1]*xi[1])/2, -(1+xi[0])*xi[1]]
            dNdxi[6] = [-xi[0]*(1+xi[1]), (1-xi[0]*xi[0])/2]
            dNdxi[7] = [-(1-xi[1]*xi[1])/2, -(1-xi[0])*xi[1]]

    # ------------------------------
    # 3D elements
    elif ndim ==3:
        if nnodes_el == 8:  # C3D8
            dNdxi[0] = [-(1-xi[1])*(1-xi[2]), -(1-xi[0])*(1-xi[2]), -(1-xi[0])*(1-xi[1])]
            dNdxi[1] = [ (1-xi[1])*(1-xi[2]), -(1+xi[0])*(1-xi[2]), -(1+xi[0])*(1-xi[1])]
            dNdxi[2] = [ (1+xi[1])*(1-xi[2]),  (1+xi[0])*(1-xi[2]), -(1+xi[0])*(1+xi[1])]
            dNdxi[3] = [-(1+xi[1])*(1-xi[2]),  (1-xi[0])*(1-xi[2]), -(1-xi[0])*(1+xi[1])]
            dNdxi[4] = [-(1-xi[1])*(1+xi[2]), -(1-xi[0])*(1+xi[2]),  (1-xi[0])*(1-xi[1])]
            dNdxi[5] = [ (1-xi[1])*(1+xi[2]), -(1+xi[0])*(1+xi[2]),  (1+xi[0])*(1-xi[1])]
            dNdxi[6] = [ (1+xi[1])*(1+xi[2]),  (1+xi[0])*(1+xi[2]),  (1+xi[0])*(1+xi[1])]
            dNdxi[7] = [-(1+xi[1])*(1+xi[2]),  (1-xi[0])*(1+xi[2]),  (1-xi[0])*(1+xi[1])]
            dNdxi = 1/8*dNdxi

    else:
        raise f'Model does not support ndim={ndim} and nnodes_el={nnodes_el}'

    return dNdxi


def integration_points_weights(nnodes_el, ndim, reduced=False):

    # Integration points sequence should be in accordance with the node sequence in element connectivity!
    if ndim == 1:  # 1D model
        if reduced:
            xi = np.array([0])
            w  = np.array([2])
        else:
            xi = np.array([[-0.5773502692], [0.5773502692]])
            w  = np.array([1, 1])

    elif ndim == 2: # 2D model
        if nnodes_el in [3, 6]:  # triangle element
            if reduced:
                xi = np.array([[1/3, 1/3]])
                w = np.array([1/2])
            else:
                xi = np.array([[0.6, 0.2], [0.2, 0.6], [0.2, 0.2]])
                w  = np.array([1/6, 1/6, 1/6])
        elif nnodes_el in [4, 8]:  # quadrilateral element
            if reduced:
                xi = np.array([[1/4, 1/4, 1/4]])
                w = np.array([1/6])
            else:
                xi = np.array([[1/4, 1/4, 1/4]])
                w = np.array([1/6])

    elif ndim ==3:  # 3D model
        if reduced:
            xi = np.array([0, 0, 0])
            w  = np.array([2])
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


if __name__ == '__main__':

    inp_fpath = './FEA sample code_Bower/Linear_elastic_quad4.txt'
    read_inp(inp_fpath, 'bower')