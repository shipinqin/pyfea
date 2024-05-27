import numpy as np


def read_inp(inp_fpath, inp_format):

    assert inp_format in ['bower', 'abq']

    model = {}
    # If the input file is ABAQUS inp style
    if inp_format == 'abq':
        nodes, elements, ndim, nnodes, nelnodes, nel = read_abq_inp(inp_fpath)

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
        nodes, elements, ndim, nnodes, nelnodes, nel, props, BCs, tractions, body_force = read_bower_inp(inp_fpath)
        if len(props) == 3:
            props = {'mu': props[0],
                     'nu': props[1],
                     'PSPE': props[2]}  # according to Bower, 0 means PS, 1 means PE
            props.update({'E': 2*props['mu']*(1+props['nu'])})

    return nodes, elements, ndim, nnodes, nelnodes, nel, props, BCs, tractions, body_force


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
        nelnodes, nel = len(elements.values[0]), len(elements)

        body_force = None

    return nodes, elements, ndim, nnodes, nelnodes, nel


def read_bower_inp(inp_fpath):

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
        BCs = np.array([BC.strip().split() for BC in BCs]).astype('float')
        BCs[:2,0] -= 1  # Python is 0-indexed
        i += nBCs

        # Traction
        ntraction = get_value(lines[i], 'int')
        i += 2

        # list of [Element_#, Face_#, Traction_components]
        tractions = lines[i:i+ntraction]
        tractions = np.array([traction.strip().split() for traction in tractions]).astype('float')
        tractions[:,:2] -= 1  # Python is 0-indexed

        # dict of [Element_#: force vector] force vector: ndim x 1
        body_force = None

    return nodes, elements, ndim, nnodes, nelnodes, nel, props, BCs, tractions, body_force


def shape_func(xi, nelnodes, ndim):

    # Shape function sequence should be in accordance with the node sequence in element connectivity!

    # xi (1 x ndim) is the local coordinate
    N = np.zeros(nelnodes)

    # ------------------------------
    # 1D elements
    if ndim == 1:
        if nelnodes == 2:  # C1D2
            N = np.array([(1-xi)/2, (1+xi)/2])
        elif nelnodes == 3:  # C1D3
            N = np.array([-xi*(1-xi)/2, xi*(1+xi)/2, (1-xi)*(1+xi)])
        return N

    # ------------------------------
    # 2D elements
    if ndim == 2:
        if nelnodes == 3:  # 1st order triangle
            N = np.array([xi[0], xi[1], 1-xi[0]-xi[1]])
        elif nelnodes == 6:  # 2nd order triangle
            xi2 = 1-xi[0]-xi[1]
            N[0] = (2*xi[0]-1)*xi[0]
            N[1] = (2*xi[1]-1)*xi[1]
            N[2] = (2*xi2-1)*xi2
            N[3] = 4*xi[0]*xi[1]
            N[4] = 4*xi[1]*xi2
            N[5] = 4*xi[0]*xi2
        elif nelnodes == 4:  # 1st order quad
            N[0] = (1-xi[0])*(1-xi[1])/4
            N[1] = (1+xi[0])*(1-xi[1])/4
            N[2] = (1+xi[0])*(1+xi[1])/4
            N[3] = (1-xi[0])*(1+xi[1])/4
        elif nelnodes == 8:  # 2nd order quad
            N[0] = -(1-xi[0])*(1-xi[1])*(1+xi[0]+xi[1])/4
            N[1] = (1+xi[0])*(1-xi[1])*(xi[0]-xi[1]-1)/4
            N[2] = (1+xi[0])*(1+xi[1])*(xi[0]+xi[1]-1)/4
            N[3] = (1-xi[0])*(1+xi[1])*(xi[0]-xi[1]-1)/4
            N[4] = (1-xi[0]*xi[0])*(1-xi[1])/2
            N[5] = (1+xi[0])*(1-xi[1]*xi[1])/2
            N[6] = (1-xi[0]*xi[0])*(1+xi[1])/2
            N[7] = (1-xi[0])*(1-xi[1]*xi[1])/2
        return N

    # ------------------------------
    # 3D elements
    if ndim == 3:
        if nelnodes == 8:  # C3D8
            for n_local in range(1,9):
                T1 = 1-xi[0] if n_local in [1, 4, 5, 8] else 1+xi[0]
                T2 = 1-xi[1] if n_local in [1, 2, 5, 6] else 1+xi[1]
                T3 = 1-xi[2] if n_local in [1, 2, 3, 4] else 1+xi[2]
                N[n_local-1] = 1/8*T1*T2*T3
        return N

    raise Exception(f'Model does not support ndim={ndim} and nelnodes={nelnodes}')


def shape_func_deriv(xi, nelnodes, ndim):

    # Shape function sequence should be in accordance with the node sequence in element connectivity!

    # x.T*shape_dev gives dx/dxi
    dNdxi = np.zeros((nelnodes, ndim))

    # ------------------------------
    # 1D elements
    if ndim == 1:
        if nelnodes == 2:  # C1D2
            dNdxi = np.array([[-1/2], [1/2]])
        elif nelnodes == 3:  # C1D3
            dNdxi = np.array([xi-1/2, xi+1/2, -2*xi])

    # ------------------------------
    # 2D elements
    elif ndim == 2:
        if nelnodes == 3:  # 1st order triangle
            dNdxi[0] = [1, 0]
            dNdxi[1] = [0, 1]
            dNdxi[2] = [-1, -1]
        elif nelnodes == 6:  # 2nd order triangle
            xi2 = 1-xi[0]-xi[1]
            dNdxi[0] = [4*xi[0]-1, 0]
            dNdxi[1] = [0, 4*xi[1]-1]
            dNdxi[2] = [-(4*xi2-1), -(4*xi2-1)]
            dNdxi[3] = [4*xi[1], 4*xi[0]]
            dNdxi[4] = [-4*xi[1], 4*(xi2-xi[1])]
            dNdxi[5] = [4*(xi2-xi[0]), -4*xi[0]]
            # In Bower's code, dNdxi[4] and dNdxi[5] are wrong
        elif nelnodes == 4:  # 1st order quad
            dNdxi[0] = [-(1-xi[1])/4, -(1-xi[0])/4]
            dNdxi[1] = [(1-xi[1])/4, -(1+xi[0])/4]
            dNdxi[2] = [(1+xi[1])/4, (1+xi[0])/4]
            dNdxi[3] = [-(1+xi[1])/4, (1-xi[0])/4]
        elif nelnodes == 8:  # 2nd order quad
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
        if nelnodes == 8:  # C3D8
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
        raise f'Model does not support ndim={ndim} and nelnodes={nelnodes}'

    return dNdxi


def integration_points_weights(nelnodes, ndim, reduced=False):

    # Integration points sequence should be in accordance with the node sequence in element connectivity!
    if ndim == 1:  # 1D model
        if reduced:
            xi = np.array([0])
            w  = np.array([2])
        else:
            xi = np.array([[-0.5773502692], [0.5773502692]])
            w  = np.array([1, 1])

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


def element_matrices(model):

    nelnodes, ndim = model.nelnodes, model.ndim
    xi_list, w = integration_points_weights(nelnodes, ndim, reduced=model.reduced_int)
    K_e = np.zeros((model.nel, nelnodes*ndim, nelnodes*ndim))
    f_e = np.zeros((model.nel, nelnodes*ndim))

    for iel, element in enumerate(model.elements):
        node_num = element
        xs_node = model.nodes[node_num]                # (nelnodes x ndim)
        for ixi, xi in enumerate(xi_list):
            # This part is performed at the integration point
            # xs_int = xs_node.T@shape_func(xi, nelnodes, ndim)    # (ndim x 1) integration point coords
            N = shape_func(xi, nelnodes, ndim)            # (nelnodes)
            dNdxi = shape_func_deriv(xi, nelnodes, ndim)            # (nelnodes x ndim)
            dxdxi = xs_node.T@dNdxi               # (ndim x ndim)
            # dxidx = np.linalg.inv(dxdxi)          # (ndim x ndim)
            # dNdx  = dNdxi@dxidx                   # (nelnodes x ndim)
            J = np.linalg.det(dxdxi)
            for inode, dNdxi_inode in enumerate(dNdxi):  # inode is the i-th node in the element
                ind_i = inode*ndim

                # Element force vector
                if model.body_force:
                    bf = model.body_force[iel]  # (ndim x 1)
                    f_e[iel, ind_i:ind_i+ndim] += bf * w[ixi]*N[inode]*J  # (ndim x 1)

                for jnode, dNdxi_jnode in enumerate(dNdxi):
                    ind_j = jnode*ndim

                    # Element stiffness matrix
                    K_e[iel, ind_i:ind_i+ndim, ind_j:ind_j+ndim] += w[ixi]/J*model.props['E']*np.outer(dNdxi_inode, dNdxi_jnode)  # (ndim x ndim)

    return K_e, f_e


def global_matrices(model, K_e, f_e):

    nnodes, ndim = model.nnodes, model.ndim
    K_global = np.zeros((nnodes*ndim, nnodes*ndim))
    f_global = np.zeros((nnodes*ndim))

    for iel, element in enumerate(model.elements):
        node_num = element
        # xs_node = model.nodes[node_num]                # (nelnodes x ndim)
        for inode, nodei in enumerate(node_num):  # nodei is the node number, inode is the i-th node in element definition
            ind_ii, ind_i = nodei*ndim, inode*ndim
            f_global[ind_ii:ind_ii+ndim] += f_e[iel, ind_i:ind_i+ndim]

            for jnode, nodej in enumerate(node_num):
                ind_jj, ind_j = nodej*ndim, jnode*ndim
                K_global[ind_ii:ind_ii+ndim, ind_jj:ind_jj+ndim] += K_e[iel, ind_i:ind_i+ndim, ind_j:ind_j+ndim]

    return K_global, f_global


def nfacenodes(ndim, nelnodes):
#====================== No. nodes on element faces ================
#
#    Copied from Bower's MatLab code
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

def get_face_nodes(ndim,nelnodes,face):
#======================= Lists of nodes on element faces =============
#
#    Copied from Bower's MatLab code
#    This procedure returns the list of nodes on an element face
#    The nodes are ordered so that the element face forms either
#    a 1D line element or a 2D surface element for 2D or 3D problems
#

    i3 = [2,3,1]
    i4 = [2,3,4,1]

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


def apply_boundary_conditions(model, K_global, f_global):

    ndim, nelnodes = model.ndim, model.nelnodes
    xi_list, w = integration_points_weights(nelnodes, ndim, reduced=model.reduced_int)

    # Update the global force vector with the traction term
    K_mod, f_mod = K_global, f_global
    for traction in model.tractions:  # [Element_#, Face_#, Traction_components]
        face_element = int(traction[0])
        element_nodes_global = model.elements[face_element]
        face = int(traction[1])
        face_nodes_local = get_face_nodes(ndim, nelnodes, face)
        face_nodes_global = element_nodes_global[face_nodes_local]
        xs_facenode = model.nodes[face_nodes_global]              # (nfacenodes x ndim)
        for xi in xi_list:
            N = shape_func(xi, nelnodes, ndim)            # (nelnodes)
            dNdxi = shape_func_deriv(xi, nelnodes, ndim)  # (nelnodes x ndim)
            N_face = N[face_nodes_local]                  # (nfacenodes)
            dxdxi_face = xs_facenode.T@dNdxi              # (ndim x ndim)
            # dxidx = np.linalg.inv(dxdxi)                # (ndim x ndim)
            # dNdx  = dNdxi@dxidx                         # (nelnodes x ndim)
            J_face = np.linalg.det(dxdxi_face)


        f_mod[face_nodes_global*ndim+] += traction[1:]

    # Displacement boundary conditions
    for BC in model.BCs:  # [Node_#, DOF#, Value]
        row = int(BC[0]*model.ndim + BC[1])
        K_mod[row,:] = 0
        K_mod[row,row] = 1
        f_mod[row] = BC[2]

    return K_mod, f_mod


if __name__ == '__main__':

    inp_fpath = './MatLab sample code_Bower/Linear_elastic_quad4.txt'
    read_inp(inp_fpath, 'bower')