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
    Return true if the hydrogen atoms are less than the 20% of the all atoms, false otherwise
    :param pdb: the protein's information
    """
    total = len(pdb.df['ATOM'])
    hydrogen_count: DataFrame = pdb.df['ATOM'][pdb.df['ATOM']['element_symbol'] == 'H']
    h = len(hydrogen_count)
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


def is_tangent(r_p, r_i, r_j, r_k, r, o_i, o_j, o_k):
    """
    Return true if the probe sphere is tangent to all the other three sphere, false otherwise
    :param r_p: the center of the probe sphere
    :param r_i: the center of the first sphere
    :param r_j: the center of the second sphere
    :param r_k: the center of the third sphere
    :param r: the radius of the probe sphere
    :param o_i: the radius of the first sphere
    :param o_j: the radius of the second sphere
    :param o_k: the radius of the third sphere
    """
    if tangent(r_p, r_i, r, o_i) and tangent(r_p, r_j, r, o_j) and tangent(r_p, r_k, r, o_k):
        return True
    return False


def probe_sphere_control(r_p, r_i, r_j, r_k, r, o_i, o_j, o_k, protein):
    """
    Return true if the probe sphere is tangent to all the other three sphere, false otherwise
    :param r_p: the center of the probe sphere
    :param r_i: the center of the first sphere
    :param r_j: the center of the second sphere
    :param r_k: the center of the third sphere
    :param r: the radius of the probe sphere
    :param o_i: the radius of the first sphere
    :param o_j: the radius of the second sphere
    :param o_k: the radius of the third sphere
    :param protein: the atom's protein
    """
    if is_tangent(r_p, r_i, r_j, r_k, r, o_i, o_j, o_k):
        for t in range(len(protein)):
            if distance(r_p[0], protein[t][0], r_p[1], protein[t][1], r_p[2], protein[t][2]) < \
                    radii[protein[t][3]] + r - 1e-5:
                return False
        return True
    return False


def find_high(r_b, r_i, r, o_i):
    """
    Return the high
    :param r_b: the point b
    :param r_i: the center of the first sphere
    :param r: the radius of the probe sphere to be found
    :param o_i: the radius of the first sphere
    """
    return (o_i + r) ** 2 - (distance(r_b[0], r_i[0], r_b[1], r_i[1], r_b[2], r_i[2])) ** 2


def find_vectors(r_i, r_j, r_k, o_i, o_j, o_k, r):
    """
    Return the T vectors
    :param r_i: the center of the first sphere
    :param r_j: the center of the second sphere
    :param r_k: the center of the third sphere
    :param r: the radius of the probe sphere
    :param o_i: the radius of the first sphere
    :param o_j: the radius of the second sphere
    :param o_k: the radius of the third sphere
    """
    t_ij = []
    t_jk = []
    t_ik = []
    for t in range(3):
        t_ij.append(1 / 2 * (r_i[t] + r_j[t]) + ((o_i + r) ** 2 - (o_j + r) ** 2) * (r_j[t] - r_i[t]) / (
                2 * (distance(r_i[0], r_j[0], r_i[1], r_j[1], r_i[2], r_j[2])) ** 2))
        t_jk.append(1 / 2 * (r_j[t] + r_k[t]) + ((o_j + r) ** 2 - (o_k + r) ** 2) * (r_k[t] - r_j[t]) / (
                2 * (distance(r_k[0], r_j[0], r_k[1], r_j[1], r_k[2], r_j[2])) ** 2))
        t_ik.append(1 / 2 * (r_i[t] + r_k[t]) + ((o_i + r) ** 2 - (o_k + r) ** 2) * (r_k[t] - r_i[t]) / (
                2 * (distance(r_k[0], r_i[0], r_k[1], r_i[1], r_k[2], r_i[2])) ** 2))

    t_ik = np.array(t_ik)
    t_ij = np.array(t_ij)
    t_jk = np.array(t_jk)
    return t_ij, t_jk, t_ik


def find_probe_sphere(r_i, r_j, r_k, o_i, o_j, o_k, r, protein):
    """
    Return the probe sphere founded
    :param r_i: the center of the first sphere
    :param r_j: the center of the second sphere
    :param r_k: the center of the third sphere
    :param r: the radius of the probe sphere
    :param o_i: the radius of the first sphere
    :param o_j: the radius of the second sphere
    :param o_k: the radius of the third sphere
    :param protein: the atom's protein
    """
    probe_sphere = []

    t_ij, t_jk, t_ik = find_vectors(r_i, r_j, r_k, o_i, o_j, o_k, r)

    r_i = np.array(r_i)
    r_j = np.array(r_j)
    r_k = np.array(r_k)

    x = r_j - r_i
    x = x / np.linalg.norm(x)
    y = r_k - r_i
    y = y - (np.dot(y, x) / np.dot(x, x)) * x
    y = y / np.linalg.norm(y)
    u = (np.dot(np.array(t_ik - t_ij), np.array(t_ik - r_i)) / np.dot(np.array(t_ik - r_i), y)) * y

    r_b = r_i + (t_ij - r_i) + u

    h = find_high(r_b, r_i, r, o_i)

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

    r_p = r_b + h * z
    if probe_sphere_control(r_p, r_i, r_j, r_k, r, o_i, o_j, o_k, protein):
        probe_sphere.append([r_p, r])

    r_p = r_b - h * z
    if probe_sphere_control(r_p, r_i, r_j, r_k, r, o_i, o_j, o_k, protein):
        probe_sphere.append([r_p, r])

    return probe_sphere


def check_distance(r_1, r_2, o_1, o_2, r):
    """
    Return true if the spheres are near enough, false otherwise
    :param r_1: the center of the first sphere
    :param r_2: the center of the second sphere
    :param o_1: the radius of the first sphere
    :param o_2: the radius of the second sphere
    :param r: the radius of the probe sphere
    :return: true if the spheres are near enough, false otherwise
    """
    if distance(r_1[0], r_2[0], r_1[1], r_2[1], r_1[2], r_2[2]) > o_1 + o_2 + 2 * r + 1e-5:
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
        r_i = [protein[i][0], protein[i][1], protein[i][2]]
        o_i = radii[protein[i][3]]
        for j in range(i + 1, len(protein)):
            r_j = [protein[j][0], protein[j][1], protein[j][2]]
            o_j = radii[protein[j][3]]
            if check_distance(r_i, r_j, o_i, o_j, r):
                continue

            for k in range(j + 1, len(protein)):
                r_k = [protein[k][0], protein[k][1], protein[k][2]]
                o_k = radii[protein[k][3]]

                if check_distance(r_i, r_k, o_i, o_k, r):
                    continue
                if check_distance(r_j, r_k, o_j, o_k, r):
                    continue

                putative_probe_sphere = find_probe_sphere(r_i, r_j, r_k, o_i, o_j, o_k, r, protein)
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
    probe_sphere = []
    for i in range(len(total_layer)):
        r_i = [total_layer[i][0][0][0], total_layer[i][0][0][1], total_layer[i][0][0][2]]
        o_i = total_layer[i][0][1]

        for j in range(i + 1, len(total_layer)):
            r_j = [total_layer[j][0][0][0], total_layer[j][0][0][1], total_layer[j][0][0][2]]
            o_j = total_layer[j][0][1]
            if check_distance(r_i, r_j, o_i, o_j, r):
                continue

            for k in range(j + 1, len(total_layer)):
                if i < total_previous_layers_length and j < total_previous_layers_length and \
                        k < total_previous_layers_length:
                    continue

                r_k = [total_layer[k][0][0][0], total_layer[k][0][0][1], total_layer[k][0][0][2]]
                o_k = total_layer[k][0][1]

                if check_distance(r_i, r_k, o_i, o_k, r):
                    continue
                if check_distance(r_j, r_k, o_j, o_k, r):
                    continue

                putative_probe_sphere = find_probe_sphere(r_i, r_j, r_k, o_i, o_j, o_k, r, protein)
                for probe in putative_probe_sphere:
                    probe_sphere.append(probe)

    return probe_sphere


def filter_not_buried_probes(current_layer, protein, bc_threshold, r):
    """
    Filter the probe spheres that are not buried enough
    :param current_layer: the layer to be filtered
    :param protein: the atom's protein
    :param bc_threshold: the threshold
    :param r : radius used to compute burial counts
    """
    buried_probes = []
    for i in range(len(current_layer)):
        count = 0
        for j in range(len(protein)):
            if distance(current_layer[i][0][0], protein[j][0], current_layer[i][0][1], protein[j][1],
                        current_layer[i][0][2], protein[j][2]) < r - 1e-5:
                count += 1

        if count >= bc_threshold:
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
                        break
                    elif current_layer[i][1] == current_layer[j][1]:
                        if i > j:
                            flag = False
                            break

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


def smooth(probe_spheres, r, minimum):
    """
    Smoothing the probe_spheres
    :param probe_spheres: all the probes found
    :param r: visualization radius
    :param minimum: the minimum number of probes nearby
    :return: the probe spheres smoothed
    """
    probes = []
    for probe1 in probe_spheres:
        count = 0
        for probe2 in probe_spheres:
            if distance(probe1[0][0][0], probe2[0][0][0], probe1[0][0][1],
                        probe2[0][0][1], probe1[0][0][2], probe2[0][0][2]) <= r + 1e-5:
                count += 1

        if count >= minimum + 1:
            probes.append(probe1)

    return probes


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


def calculate_pw(probe_spheres, d, r):
    """
    Calculate the PW of each probe sphere
    :param probe_spheres: the final probe spheres
    :param d: parameter defining the probe weight (PW) envelope function
    :param r: parameter defining the probe weight (PW) envelope function
    :return: the list of probe spheres with the PW weight of each one of them
    """
    pw_probes = []
    for i in probe_spheres:
        pw = 0
        for j in probe_spheres:
            pw += j[1] * math.exp(-
                                  (distance(i[0][0][0], j[0][0][0], i[0][0][1], j[0][0][1], i[0][0][2],
                                            j[0][0][2]) - r) ** 2 / d ** 2)
        pw_probes.append((i, pw))
    return pw_probes


def take_pw(elem):
    """
    Return the PW weight of the given probe sphere
    :param elem: a probe sphere
    :return: the PW weight
    """
    return elem[1]


def find_asp_points(probe_spheres, r, minimum):
    """
    Find the ASP points
    :param probe_spheres: all the probes founded
    :param r: ASP radius, the minimum distance between two ASP points
    :param minimum: the minimum PW of ASP points
    :return: the ASP points
    """
    asp_point = []
    for elem in probe_spheres:
        if elem[1] >= minimum:
            flag = True
            for point in asp_point:
                if distance(elem[0][0][0][0], point[0][0][0], elem[0][0][0][1], point[0][0][1], elem[0][0][0][2],
                            point[0][0][2]) < r:
                    flag = False
                    break
            if flag:
                asp_point.append(elem[0])
    return asp_point


# ###########
# Entry point
# ###########

if __name__ == '__main__':
    from sys import argv

    # ###########
    # Protein and output paths
    # ###########
    if len(argv) > 3:
        path_input = argv[1]
        path_output = argv[2]
        path_asp = argv[3]
    else:
        print("Insert the path of the input, the path of the output (probe spheres) and the path of the ASP")
        exit(1)

    if not path_input.endswith('.pdb'):
        print("The input path must be .pdb")
        exit(1)
    if not path_output.endswith('.pdb'):
        print("The output path for the probe spheres must be .pdb")
        exit(1)
    if not path_asp.endswith('.pdb'):
        print("The output path for the ASP must be .pdb")
        exit(1)

    # ###########
    # Parameters
    # ###########
    r_bc = 8.0
    r_weed = 1.0
    r_accretion = 0.7
    r_vis = 2.5
    minimum_probe = 4
    r_o = 2.0
    d_o = 1.0
    r_asp = 8.0
    pw_min = 1100
    s = True

    # Smooth
    if len(argv) > 4:
        if argv[4] == "-all":
            s = False

    # Read the file PDB
    try:
        pdb1 = reading_file(path_input)
    except ValueError or FileNotFoundError:
        print("The input path is wrong")
        exit(1)

    # Remove the hydrogen based on their percentage in the all protein
    removing_hydrogen = hydrogen(pdb1)
    atoms = delete_hydrogen(removing_hydrogen, pdb1)
    # Get the r_probe
    r_probe = find_radius(removing_hydrogen)
    # Get the BC threshold
    bc = find_bc_threshold(removing_hydrogen)

    # ###########
    # Construction of the first layer
    # ###########

    # Build the first layer
    c = initial_layer(atoms, r_probe)
    # Filter the non-buried probes
    filtered_1 = filter_not_buried_probes(c, atoms, bc, r_bc)
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
        filtered_1 = filter_not_buried_probes(filtered_0, atoms, bc, r_bc)
        # Filter the probes in order to let them be more spread
        filtered_2 = filter_not_distributed_probes(filtered_1, r_weed)

    # Smooth the probe spheres
    all_probes = result
    if s:
        result = smooth(result, r_vis, minimum_probe)

    # Create the file PDB for the probe spheres
    if len(result) > 0:
        layer = atoms_list(result)
        layer = from_data_frame_to_pdb(layer)
        layer.to_pdb(path=path_output,
                     records=['ATOM'],
                     gz=False,
                     append_newline=True)
    else:
        print("There are not probe spheres")

    # Calculate the probes weight
    asp = calculate_pw(all_probes, d_o, r_o)
    # Sort the probe spheres based on the PW
    asp.sort(key=take_pw, reverse=True)
    # Find the asp points
    asp = find_asp_points(asp, r_asp, pw_min)

    # Create the file PDB for the asp points
    if len(asp) > 0:
        layer = atoms_list(asp)
        layer = from_data_frame_to_pdb(layer)
        layer.to_pdb(path=path_asp,
                     records=['ATOM'],
                     gz=False,
                     append_newline=True)
    else:
        print("There are not asp points")
