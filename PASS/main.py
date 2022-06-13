import math
import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from pandas import DataFrame

# Radius of each atom
radii = {
    'N': 1.55,
    'O': 1.52,
    'C': 1.70,
    'F': 1.80,
    'S': 1.80,
    'H': 1.20,
}


def reading_file(protein):
    """
    Read the PDB file
    """
    pdb: PandasPdb = PandasPdb().read_pdb(protein)
    return pdb


def hydrogen(pdb):
    """
     Return true if the hydrogen atoms are less than the 20% of the all atoms
     :param pdb: the protein's information
     """
    total = len(pdb.df['ATOM'])
    no_hydrogen: DataFrame = pdb.df['ATOM'][pdb.df['ATOM']['element_symbol'] == 'H']
    h = len(no_hydrogen)
    if h <= 0.2 * total:
        return True
    return False


def delete_hydrogen(value, pdb):
    """
     Return the protein without the hydrogen atoms if the parameter x is true
     :param value: true if hydrogen atoms are less than the 20% of all atoms
     :param pdb: PandasPdb
     """
    protein_atoms = pdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord', 'element_symbol']]
    if value:
        protein_atoms = protein_atoms[protein_atoms.element_symbol != 'H']
    return protein_atoms.to_numpy()


def find_radius(value):
    """
     Return the radius of the probe spheres
     :param value: true if hydrogen atoms are less than the 20% of all atoms
     """
    if value:
        return 1.8
    return 1.5


def find_bc_threshold(value):
    """
     Return the BC value
     :param value: true if hydrogen atoms are less than the 20% of all atoms
     """
    if value:
        return 55
    return 75


def distance(x1, x2, y1, y2, z1, z2):
    """
     Return the 3D distance between two points
     :param x1: first x-coordinate
     :param x2: second x-coordinate
     :param y1: first y-coordinate
     :param y2: second y-coordinate
     :param z1: first z-coordinate
     :param z2: second z-coordinate
     """
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


def tangent(c1, c2, r1, r2):
    """
     Return true two spheres are tangent, false otherwise
     :param c1: the center of the first sphere
     :param c2: the center of the second sphere
     :param r1: the radius of the first sphere
     :param r2: the radius of the second sphere
     """
    if distance(c1[0], c2[0], c1[1], c2[1], c1[2], c2[2]) < (r1 + r2) * (1 - 1e-5):
        return False
    if distance(c1[0], c2[0], c1[1], c2[1], c1[2], c2[2]) > (r1 + r2) * (1 + 1e-5):
        return False

    return True


def is_tangent(R_p, R_i, R_j, R_k, r, o_i, o_j, o_k):
    """
     Return true if the probe sphere is tangent to all the other three sphere, false otherwise
     :param R_p: the center of the probe sphere
     :param R_i: the center of the first sphere
     :param R_j: the center of the second sphere
     :param R_k: the center of the third sphere
     :param r: the radius of the probe sphere
     :param o_i: the radius of the first sphere
     :param o_j: the radius of the second sphere
     :param o_k: the radius of the third sphere
     """
    if tangent(R_p, R_i, r, o_i) and tangent(R_p, R_j, r, o_j) and tangent(R_p, R_k, r, o_k):
        return True
    return False


def probe_sphere_control(R_p, R_i, R_j, R_k, r, o_i, o_j, o_k, protein):
    """
     Return true if the probe sphere is tangent to all the other three sphere, false otherwise
     :param R_p: the center of the probe sphere
     :param R_i: the center of the first sphere
     :param R_j: the center of the second sphere
     :param R_k: the center of the third sphere
     :param r: the radius of the probe sphere
     :param o_i: the radius of the first sphere
     :param o_j: the radius of the second sphere
     :param o_k: the radius of the third sphere
     :param protein: the atom's protein
     """
    if is_tangent(R_p, R_i, R_j, R_k, r, o_i, o_j, o_k):
        for t in range(len(protein)):
            if distance(R_p[0], protein[t][0], R_p[1], protein[t][1], R_p[2], protein[t][2]) < \
                    radii[protein[t][3]] + r - 1e-5:
                return False
        return True
    return False


def find_high(R_b, R_i, r, o_i):
    """
     Return the high
     :param R_b: the point b
     :param R_i: the center of the first sphere
     :param r: the radius of the probe sphere to be found
     :param o_i: the radius of the first sphere
     """
    return (o_i + r) ** 2 - (distance(R_b[0], R_i[0], R_b[1], R_i[1], R_b[2], R_i[2])) ** 2


def find_vectors(R_i, R_j, R_k, o_i, o_j, o_k, r):
    """
     Return the T vectors
     :param R_i: the center of the first sphere
     :param R_j: the center of the second sphere
     :param R_k: the center of the third sphere
     :param r: the radius of the probe sphere
     :param o_i: the radius of the first sphere
     :param o_j: the radius of the second sphere
     :param o_k: the radius of the third sphere
     """
    T_ij = []
    T_jk = []
    T_ik = []
    for t in range(3):
        T_ij.append(1 / 2 * (R_i[t] + R_j[t]) + ((o_i + r) ** 2 - (o_j + r) ** 2) * (R_j[t] - R_i[t]) / (
                2 * (distance(R_i[0], R_j[0], R_i[1], R_j[1], R_i[2], R_j[2])) ** 2))
        T_jk.append(1 / 2 * (R_j[t] + R_k[t]) + ((o_j + r) ** 2 - (o_k + r) ** 2) * (R_k[t] - R_j[t]) / (
                2 * (distance(R_k[0], R_j[0], R_k[1], R_j[1], R_k[2], R_j[2])) ** 2))
        T_ik.append(1 / 2 * (R_i[t] + R_k[t]) + ((o_i + r) ** 2 - (o_k + r) ** 2) * (R_k[t] - R_i[t]) / (
                2 * (distance(R_k[0], R_i[0], R_k[1], R_i[1], R_k[2], R_i[2])) ** 2))

    T_ik = np.array(T_ik)
    T_ij = np.array(T_ij)
    T_jk = np.array(T_jk)
    return T_ij, T_jk, T_ik


def find_probe_sphere(R_i, R_j, R_k, o_i, o_j, o_k, r, protein):
    """
     Return the probe sphere founded
     :param R_i: the center of the first sphere
     :param R_j: the center of the second sphere
     :param R_k: the center of the third sphere
     :param r: the radius of the probe sphere
     :param o_i: the radius of the first sphere
     :param o_j: the radius of the second sphere
     :param o_k: the radius of the third sphere
     :param protein: the atom's protein
     """
    probe_sphere = []

    T_ij, T_jk, T_ik = find_vectors(R_i, R_j, R_k, o_i, o_j, o_k, r)

    R_i = np.array(R_i)
    R_j = np.array(R_j)
    R_k = np.array(R_k)

    x = R_j - R_i
    x = x / np.linalg.norm(x)
    y = R_k - R_i
    y = y - (np.dot(y, x) / np.dot(x, x)) * x
    y = y / np.linalg.norm(y)
    U = (np.dot(np.array(T_ik - T_ij), np.array(T_ik - R_i)) / np.dot(np.array(T_ik - R_i), y)) * y

    R_b = R_i + (T_ij - R_i) + U

    h = find_high(R_b, R_i, r, o_i)

    if h >= 0:
        h = math.sqrt(h)
    else:
        return probe_sphere

    z = np.random.rand(3)
    z = z - (np.dot(z, x) / np.dot(x, x)) * x - (np.dot(z, y) / np.dot(y, y)) * y

    while np.dot(z, z) < 1e-9:
        z = np.random.rand(3)
        z = z - (np.dot(z, x) / np.dot(x, x)) * x - (np.dot(z, y) / np.dot(y, y)) * y

    z = z / np.linalg.norm(z)

    R_p = R_b + h * z
    if probe_sphere_control(R_p, R_i, R_j, R_k, r, o_i, o_j, o_k, protein):
        probe_sphere.append([R_p, r])

    R_p = R_b - h * z
    if probe_sphere_control(R_p, R_i, R_j, R_k, r, o_i, o_j, o_k, protein):
        probe_sphere.append([R_p, r])

    return probe_sphere


def check_distance(R_i, R_j, R_k, o_i, o_j, o_k, r):
    if distance(R_i[0], R_j[0], R_i[1], R_j[1], R_i[2], R_j[2]) > o_i + o_j + 2 * r:
        return True
    if distance(R_i[0], R_k[0], R_i[1], R_k[1], R_i[2], R_k[2]) > o_i + o_k + 2 * r:
        return True
    if distance(R_j[0], R_k[0], R_j[1], R_k[1], R_j[2], R_k[2]) > o_j + o_k + 2 * r:
        return True
    return False


def initial_layer(protein, r):
    """
     Create the first layer
     :param protein: the atom's protein
     :param r: the radius of the probe spheres created
     """
    probe_sphere = []
    for i in range(len(protein)):
        for j in range(i + 1, len(protein)):
            for k in range(j + 1, len(protein)):
                R_i = [protein[i][0], protein[i][1], protein[i][2]]
                R_j = [protein[j][0], protein[j][1], protein[j][2]]
                R_k = [protein[k][0], protein[k][1], protein[k][2]]
                o_i = radii[protein[i][3]]
                o_j = radii[protein[j][3]]
                o_k = radii[protein[k][3]]

                if check_distance(R_i, R_j, R_k, o_i, o_j, o_k, r):
                    continue

                putative_probe_sphere = find_probe_sphere(R_i, R_j, R_k, o_i, o_j, o_k, r, protein)
                for probe in putative_probe_sphere:
                    probe_sphere.append(probe)

    return probe_sphere


def accretion_layer(protein, total_layer, r, total_previous_layers_length):
    """
     Create the following layers
     :param protein: the atom's protein
     :param total_layer: all probe spheres
     :param r: the radius of the probe spheres created
     :param total_previous_layers_length: the length of the previous layers
     """
    print("Total computation:")
    print(len(total_layer))

    probe_sphere = []
    for i in range(len(total_layer)):
        for j in range(i + 1, len(total_layer)):
            for k in range(j + 1, len(total_layer)):
                if i < total_previous_layers_length and j < total_previous_layers_length and \
                        k < total_previous_layers_length:
                    continue

                R_i = [total_layer[i][0][0][0], total_layer[i][0][0][1], total_layer[i][0][0][2]]
                R_j = [total_layer[j][0][0][0], total_layer[j][0][0][1], total_layer[j][0][0][2]]
                R_k = [total_layer[k][0][0][0], total_layer[k][0][0][1], total_layer[k][0][0][2]]
                o_i = total_layer[i][0][1]
                o_j = total_layer[j][0][1]
                o_k = total_layer[k][0][1]

                if check_distance(R_i, R_j, R_k, o_i, o_j, o_k, r):
                    continue

                putative_probe_sphere = find_probe_sphere(R_i, R_j, R_k, o_i, o_j, o_k, r, protein)
                for probe in putative_probe_sphere:
                    probe_sphere.append(probe)

    return probe_sphere


def filter_not_buried_probes(current_layer, protein, BC_threshold, r):
    """
     Filter the probe spheres that are not buried enough
     :param current_layer: the layer to be filtered
     :param protein: the atom's protein
     :param BC_threshold: the threshold
     :param r : radius used to compute burial counts
     """
    # Parameter that choose how to count the proteins
    # If non_strict = 1, then all the atom's protein with distance 8 will be counted
    # If non_strict = 0, then only the atom's protein with the center distance less than 8 will be counted
    non_strict = 0
    buried_probes = []
    for i in range(len(current_layer)):
        count = 0
        for j in range(len(protein)):
            if distance(current_layer[i][0][0], protein[j][0], current_layer[i][0][1], protein[j][1],
                        current_layer[i][0][2], protein[j][2]) - non_strict * radii[protein[j][3]] < r - 1e-5:
                count += 1

        if count >= BC_threshold:
            buried_probes.append((current_layer[i], count))

    return buried_probes


def filter_not_distributed_probes(current_layer, r):
    """
     Filter the probe spheres that are not distributed enough
     :param current_layer: the layer to be filtered
     :param r: minimal separation between probe spheres
     """
    distributed_probe_spheres = []
    for i in range(len(current_layer)):
        flag = True
        for j in range(len(current_layer)):
            if i != j:
                if distance(current_layer[i][0][0][0], current_layer[j][0][0][0], current_layer[i][0][0][1],
                            current_layer[j][0][0][1],
                            current_layer[i][0][0][2], current_layer[j][0][0][2]) <= r - 1e-5:
                    if current_layer[i][1] < current_layer[j][1]:
                        flag = False
                    elif current_layer[i][1] == current_layer[j][1]:
                        if i > j:
                            flag = False

        if flag:
            distributed_probe_spheres.append(current_layer[i])

    return distributed_probe_spheres


def filter_non_distributed_probes_with_previous_layer(current_layer, previous_layer, r):
    """
     Filter the probe spheres that are not distributed enough
     :param current_layer: the layer to be filtered
     :param previous_layer: the previous layer between probe spheres
     :param r: minimal separation between probe spheres
     """
    distributed_probes = []
    print(previous_layer)
    for i in range(len(current_layer)):
        flag = True
        for j in range(len(previous_layer)):
            if distance(current_layer[i][0][0], previous_layer[j][0][0][0], current_layer[i][0][1],
                        previous_layer[j][0][0][1], current_layer[i][0][2],
                        previous_layer[j][0][0][2]) <= r - 1e-15:
                flag = False
                break

        if flag:
            distributed_probes.append(current_layer[i])

    return distributed_probes


def atoms_list(probes):
    """
     Create the atoms list for the PDB file
     :param probes: the pockets
     """
    probe_spheres = []
    for p in range(len(probes)):
        probe_spheres.append(('ATOM', p, '', 'H', '', 'ILE', '', "", p, '', '', probes[p][0][0][0], probes[p][0][0][1],
                              probes[p][0][0][2], 1.0, 0, '', '', 'H', 0, p))
    return probe_spheres


def from_data_frame_to_pdb(list1):
    """
     Create the file PDB from the atoms list
     :param list1: the atoms list
     """
    pp: DataFrame = pd.DataFrame(list1,
                                 columns=['record_name', 'atom_number', 'blank_1', 'atom_name', 'alt_loc',
                                          'residue_name', 'blank_2',
                                          'chain_id', 'residue_number', 'insertion', 'blank_3', 'x_coord', 'y_coord',
                                          'z_coord',
                                          'occupancy', 'b_factor', 'blank_4', 'segment_id', 'element_symbol', 'charge',
                                          'line_idx'])

    pdb = PandasPdb()
    pdb.df['ATOM'] = pp

    return pdb


# ###########
# Entry point
# ###########

if __name__ == '__main__':
    # ###########
    # Protein
    # ###########
    path = "1lqd_pocket.pdb"

    # ###########
    # Parameters
    # ###########
    r_bc = 8.0
    r_weed = 1.0
    r_accretion = 0.7

    # Read the file PDB
    pdb1 = reading_file(path)
    # Remove the hydrogen based on their percentage in the all protein
    removing_hydrogen = hydrogen(pdb1)
    atoms = delete_hydrogen(removing_hydrogen, pdb1)
    # Get the r_probe
    r_probe = find_radius(removing_hydrogen)
    # Get the BC threshold
    BC = find_bc_threshold(removing_hydrogen)

    # ###########
    # Construction of the first layer
    # ###########

    # Build the first layer
    c = initial_layer(atoms, r_probe)
    # Filter the non-buried probes
    filtered_1 = filter_not_buried_probes(c, atoms, BC, r_bc)
    # Filter the probes in order to let them be more spread
    filtered_2 = filter_not_distributed_probes(filtered_1, r_weed)

    # ###########
    # Construction of the other layers
    # ###########
    result = []
    while len(filtered_2) > 0:
        previous_layers_length = len(result)

        for el in filtered_2:
            result.append(el)

        # Build a layer
        c = accretion_layer(atoms, result, r_accretion, previous_layers_length)
        # Filter the probes non-distributed enough with the previous layers
        filtered_0 = filter_non_distributed_probes_with_previous_layer(c, result, r_weed)
        # Filter the non-buried probes
        filtered_1 = filter_not_buried_probes(filtered_0, atoms, BC, r_bc)
        # Filter the probes in order to let them be more spread
        filtered_2 = filter_not_distributed_probes(filtered_1, r_weed)

    # Create the file pdb
    layer = atoms_list(result)
    layer = from_data_frame_to_pdb(layer)
    layer.to_pdb(path='./output.pdb',
                 records=['ATOM'],
                 gz=False,
                 append_newline=True)
