import math
import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from numba import jit
from pandas import DataFrame

radii = {
    'N': 1.75,
    'O': 1.55,
    'C': 2.05,
    'F': 2.10,
    'S': 2.10,
    'H': 1,
}

@jit(nopython=True)
def distance(x1, x2, y1, y2, z1, z2):
    """
    Calculate 3D distance between two points
    :param x1: x coordinate of the first point
    :param x2: x coordinate of the second point
    :param y1: y coordinate of the first point
    :param y2: y coordinate of the second point
    :param z1: z coordinate of the first point
    :param z2: z coordinate of the second point
    :return: 3D distance between two points
    """
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

def reading_file(path):
    """
    Read the PDB file
    """
    ppdb: PandasPdb = PandasPdb().read_pdb(path)
    return ppdb

def delete_hydrogen(ppdb):
    """
    Return the protein without the hydrogen atoms
    :param pdb: PandasPdb
    """
    ppdb.df['ATOM'] = ppdb.df['ATOM'][ppdb.df['ATOM']['element_symbol'] != 'H']
    return ppdb

def matrix_creation(ppdb):
    """
    Creates the matrix containing the centers of the atoms with their associated element
    :param ppdb: PandasPdb
    :return: coordinates of centers and atomic element of atoms in input
    """
    matrix = ppdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord', 'element_symbol']].to_numpy()
    return matrix

def maximum_coordinate(coordinate_position, matrix, scale):
    """
    find the maximum coordinate for a specific coordinate
    :param coordinate_position: indicate for witch coordinate calculate the maximum value
    :param matrix: contains all the atoms
    :param scale: scale of the grid leg's
    :return: maximum value for the specified coordinate multiplied by scale
    """
    max_c = matrix[0][coordinate_position]
    for i in range(1, len(matrix)):
        max_c = max(max_c, matrix[i][coordinate_position])
    return max_c

def minimum_coordinate(coordinate_position, matrix, scale):
    """
    find the minimum coordinate for a specific coordinate
    :param coordinate_position: indicate for witch coordinate calculate the minimum value
    :param matrix: contains all the atoms
    :param scale: scale of the grid leg's
    :return: minimum value for the specified coordinate multiplied by scale
    """
    min_c = matrix[0][coordinate_position]
    for i in range(1, len(matrix)):
        min_c = min(min_c, matrix[i][coordinate_position])
    return min_c

def matrix_initialization(matrix, coordinates_scale):
    """
    Translate the atoms center to have all value stored in a matrix
    :param matrix: atoms centers
    :param coordinates_scale: value to scale for each coordinate
    :return: matrix with each value translated
    """
    for i in range(len(matrix)):
        for coordinate in range(len(coordinates_scale)):
            matrix[i][coordinate] -= coordinates_scale[coordinate]
    return matrix

def grid_initialization(matrix, dim, scale):
    """
    Create the grid and store it in a matrix
    :param matrix: atoms centers
    :param dim: dimesion of the grid
    :param scale: scale of the grid leg's
    :return: grid stores in a matrix with values one and zero
    """
    # Create a grid full of zeros
    dim_x = int(dim[0])
    dim_y = int(dim[1])
    dim_z = int(dim[2])
    grid = np.zeros((dim_x + 1, dim_y + 1, dim_z + 1))

    for i in range(len(matrix)):
        intx = int(matrix[i][0])
        inty = int(matrix[i][1])
        intz = int(matrix[i][2])
        atom_radious = 1.8
        if matrix[i][3] in radii:
            atom_radious = radii[matrix[i][3]]
        else:
            print('Atomo ' + matrix[i][3] + ' non esistente')
        maximum_radious = 0
        for key in radii:
            maximum_radious = max(maximum_radious, radii[key])
        maximum_radious = int(maximum_radious)
        # Points (j, k, t) are the points that can potentially be touched by an atom
        dim_x = int(dim[0])
        dim_y = int(dim[1])
        dim_z = int(dim[2])
        for j in range(max(0, (intx - (maximum_radious + 2)) * scale), min(dim_x + 1, (intx + (maximum_radious + 2)) * scale + 1)):
            for k in range(max(0, (inty - (maximum_radious + 2)) * scale), min(dim_y + 1, (inty + (maximum_radious + 2)) * scale + 1)):
                for t in range(max(0, (intz - (maximum_radious + 2)) * scale), min(dim_z + 1, (intz + (maximum_radious + 2)) * scale + 1)):
                    if distance(matrix[i][0], j / scale, matrix[i][1], k / scale, matrix[i][2], t / scale) < atom_radious:
                        grid[j][k][t] = 1

    return grid

@jit(nopython=True)
def belong_to_board(x, y, z, r, grid, dim, scale):
    """
    Find if in a center can be create a sphere that doesn't touch the protein
    :param x: x coordinate of the center
    :param y: y coordinate of the center
    :param z: z coordinate of the center
    :param r: radius of the probe sphere
    :param grid: grid containing the protein
    :param dim: dimension of grid
    :param scale: scale of the grid leg's
    :return: 0 if the probe sphere belong to the board, 1 otherwise
    """
    dim_x = int(dim[0])
    dim_y = int(dim[1])
    dim_z = int(dim[2])
    for i in range(max(0, (x - r * scale), min(dim_x + 1, x + r * scale + 1))):
        for j in range(max(0, y - r * scale), min(dim_y + 1, y + r * scale + 1)):
            for k in range(max(0, z - r * scale), min(dim_z + 1, z + r * scale + 1)):
                if grid[i][j][k] == 1 and distance(x / scale, i / scale, y / scale, j / scale, z / scale, k / scale) < r:
                    return 0
    return 1

@jit(nopython=True)
def board_creation(grid, r, dim, scale):
    """
    For each point that belong to the board, all the points contained in the sphere are set to "-1"
    :param grid: grid containing the protein
    :param r: radius of the probe sphere
    :param dim: dimension of grid
    :param scale: scale of the grid leg's
    :return: grid updated with the board identified by valie "-1"
    """
    dim_x = int(dim[0])
    dim_y = int(dim[1])
    dim_z = int(dim[2])
    for i in range(dim_x + 1):
        for j in range(dim_y + 1):
            for k in range(dim_z + 1):
                if belong_to_board(i, j, k, r, grid, dim, scale):
                    for t in range(max(0, i - r * scale), min(dim_x + 1, i + r * scale + 1)):
                        for v in range(max(0, j - r * scale), min(dim_y + 1, j + r * scale + 1)):
                            for u in range(max(0, k - r * scale), min(dim_z + 1, k + r * scale + 1)):
                                if distance(i / scale, t / scale, j /scale, v / scale, k / scale, u / scale) < r:
                                    grid[t][v][u] = -1
    return grid

@jit(nopython=True)
def delete_noise_points(grid, dim, spf):
    """
    For each point that could constitute a binding site, countspf parameter is defined
    which counts the number of points belonging to the binding site around the point considered
    and if the value just calculated is lower than the threshold value it's eliminated from the
    points belonging to the binding sites
    :param grid: grid containing the protein
    :param dim: dimension of grid
    :param spf: SPF parameter in input
    :return: grid after the deletion of noise points
    """
    dim_x = int(dim[0])
    dim_y = int(dim[1])
    dim_z = int(dim[2])
    grid_copy = np.zeros((dim_x + 1, dim_y + 1, dim_z + 1))
    for i in range(dim_x + 1):
        for j in range(dim_y + 1):
            for k in range(dim_z + 1):
                grid_copy[i][j][k] = grid[i][j][k]
    for i in range(1, dim_x):
        for j in range(1, dim_y):
            for k in range(1, dim_z):
                if grid_copy[i][j][k] != 0:
                    continue
                countspf = 0
                for c1 in range(i - 1, i + 2):
                    for c2 in range(j - 1, j + 2):
                        for c3 in range(k - 1, k + 2):
                            if c1 == i and c2 == j and c3 == k:
                                continue
                            if grid_copy[c1][c2][c3] == 0:
                                countspf += 1
                if countspf <= spf:
                    grid[i][j][k] = -2
    return grid

jit(nopython=True)
def bfs(grid, dim):
    """
    Associate each group of binding sites with a value that will then become a letter.
    It is used to identify the different groups
    :param grid: grid containing the protein
    :param dim: dimension of grid
    :return: grid with each binding sites group marked by a value
    """
    queue = []
    counter = 1
    dim_x = int(dim[0])
    dim_y = int(dim[1])
    dim_z = int(dim[2])
    for i in range(dim_x + 1):
        for j in range(dim_y + 1):
            for k in range(dim_z + 1):
                if(grid[i][j][k] == 0):
                    counter += 1
                    grid[i][j][k] = counter
                    queue.append((i, j, k))
                while(len(queue) != 0):
                    h = queue.pop()
                    x = h[0]
                    y = h[1]
                    z = h[2]
                    for c1 in range(max(0, x - 1), min(dim_x + 1, x + 2)):
                        for c2 in range(max(0, y - 1), min(dim_y + 1, y + 2)):
                            for c3 in range(max(0, z-1), min(dim_z + 1, z + 2)):
                                if grid[c1][c2][c3] == 0:
                                    grid[c1][c2][c3] = counter
                                    queue.append((c1, c2, c3))
    return grid

def distance_from_board(x, y, z, grid, dim):
    """
    Calculate the distance from the board for a point belonging to a binding sites
    :param x: x coordinate of the point
    :param y: y coordinate of the point
    :param z: z coordinate of the point
    :param grid: grid containing the protein
    :param dim: dimension of grid
    :return: distance from the board for this point
    """
    # Defines the minimum distance to "1"
    board_distance = 1
    while(True):
        dim_x = int(dim[0])
        dim_y = int(dim[1])
        dim_z = int(dim[2])
        for i in range(max(0, x - board_distance), min(dim_x + 1, x + board_distance + 1)):
            for j in range(max(0, y - board_distance), min(dim_y + 1, y + board_distance + 1)):
                for k in range(max(0, z - board_distance), min(dim_z + 1, z + board_distance + 1)):
                    if grid[i][j][k] == -1:
                        return board_distance
        board_distance += 1

def counting_sort(list1, groups_dim, grid, dim, scale, ranking_value):
    """
    Sort the elements of the list using Counting Sort algorithm
    :param list1: list to sort
    :param groups_dim: dimension of the list
    :param grid: grid containing the protein
    :param dim: dimension of grid
    :param scale: scale of the grid leg's
    :param ranking_value: true if sort groups considering each point with the Manhattan distance from the board, false considering each point with value "1"
    :return: list sorted
    """
    groups_dim = int(groups_dim)
    counting = np.zeros((groups_dim + 1))
    for i in range(len(list1)):
        distance = 1
        if ranking_value:
            distance = distance_from_board(int(list1[i][1] * scale), int(list1[i][2] * scale), int(list1[i][3] * scale), grid, dim)
        counting[list1[i][0]] += distance
    copy_counting = []
    for i in range(2, groups_dim + 1):
        copy_counting.append(counting[i])
    copy_counting.sort()
    copy_counting.reverse()
    used = np.zeros((groups_dim + 1))
    list2 = []
    for i in range(len(copy_counting)):
        current_value = 0
        for j in range(2, groups_dim + 1):
            if(counting[j] == copy_counting[i] and used[j] == 0):
                used[j] = 1
                current_value = j
                break
        for j in range(len(list1)):
            if list1[j][0] == current_value:
                list2.append(list1[j])
    return list2

def binding_sites_list_creation(grid, min_c, dim, scale, Top_n, ranking_value):
    """
    Creation of the PDB file containing the binding sites
    :param grid:
    :param min_c:
    :param dim:
    :param scale:
    :param Top_n:
    :param ranking_value:
    :return: list containing binding sites
    """
    list1 = []
    letters ="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    sites_dimension = 2
    dim_x = int(dim[0])
    dim_y = int(dim[1])
    dim_z = int(dim[2])
    for i in range(dim_x + 1):
        for j in range(dim_y + 1):
            for k in range(dim_z + 1):
                if grid[i][j][k] >= 2:
                    list1.append((int(grid[i][j][k]), i / scale + min_c[0], j / scale + min_c[1], k / scale + min_c[2]))
                    sites_dimension = max(sites_dimension, (grid[i][j][k]))
    # The list of binding sites groups is sorted in descending order according to their size
    list1 = counting_sort(list1, sites_dimension, grid, dim, scale, ranking_value)
    list2 = []
    count = 1
    current_letter = 0
    # Each point belonging to a binding site is added in list2
    for i in range(len(list1)):
        if(i!=0 and list1[i][0] != list1[i-1][0]):
            current_letter += 1
        if current_letter == Top_n:
            break
        list2.append(('ATOM', count, '', 'H', '', 'ILE', '', letters[current_letter], count, '', '', float(list1[i][1]) , float(list1[i][2]), float(list1[i][3]),
                    1.0 , 0,'', '', 'H', 0, count))
        count += 1
    return list2

def fromdataframe_topdb(list1):
    """
    The list containing all binding site points is used to create the PDB file
    :param list1: list containing binding sites to print
    :return: PDB file to print
    """
    pp: DataFrame = pd.DataFrame(list1,
                                 columns=['record_name', 'atom_number', 'blank_1', 'atom_name', 'alt_loc',
                                          'residue_name', 'blank_2',
                                          'chain_id', 'residue_number', 'insertion', 'blank_3', 'x_coord', 'y_coord',
                                          'z_coord',
                                          'occupancy', 'b_factor', 'blank_4', 'segment_id', 'element_symbol', 'charge',
                                          'line_idx'])
    ppdb = PandasPdb()
    ppdb.df['ATOM'] = pp
    return ppdb

# ###########
# Entry point
# ###########

if __name__ == '__main__':
    from sys import argv

# Leg dimension of the grid
leg = argv[1]
leg = float(leg)

# protein's path
path = argv[2]

# Radius
r = argv[3]
r = int(r)

# SPF
spf = argv[4]
spf = int(spf)

# Top_n input
Top_n = argv[5]
Top_n = int(Top_n)

# Ranking
ranking = argv[6]

if len(argv) <= 6:
    print("Insert the dimension of the grid(1 or 0.5), the path of the input(ending with .pdb), radius value(between 1 and 8), SPF value(between 0 and 27, Top_n value(between 0 and 26), Ranking value(yes or no)")
    exit(1)

if leg != 1 and leg != 0.5:
    print("Grid dimension must be 1 or 0.5")
    exit(1)

if not path.endswith('.pdb'):
    print("The input path must be .pdb")
    exit(1)

if r < 1:
    print("Radius of probe sphere must be greater than 1")
    exit(1)

if spf < 0 and spf > 27:
    print("SPF value must be between 0 and 27")
    exit(1)

if Top_n < 0 and Top_n > 27:
    print("Top_n value must be between 0 and 27")
    exit(1)

if ranking != 'no' and ranking != 'yes':
    print("Radius of probe sphere must be greater than 1")
    exit(1)

scale = 1
if leg == 0.5:
    scale = 2

ppdb1 = reading_file(path)
ppdb1 = delete_hydrogen(ppdb1)

matrix = matrix_creation(ppdb1)
# Find maximum and minimum value of atom's coordinate
max_c = np.zeros((3))
min_c = np.zeros((3))
for i in range(len(max_c)):
    max_c[i] = maximum_coordinate(i, matrix, scale)
    max_c[i] += 25 * scale
    max_c[i] = int(max_c[i])
    min_c[i] = minimum_coordinate(i, matrix, scale)
    min_c[i] -= 25 * scale
    min_c[i] = int(min_c[i])
# translate the atoms center
matrix = matrix_initialization(matrix, min_c)
dim = np.zeros((3))
for i in range(3):
    dim[i] = max_c[i] - min_c[i] + max(50 * scale, 4 * r * scale)
grid = grid_initialization(matrix, dim, scale)

ranking_value = False
if ranking == 'yes':
    ranking_value = True

grid = board_creation(grid, r, dim, scale)
grid = delete_noise_points(grid, dim, spf)
grid = bfs(grid, dim)
binding_sites_list = binding_sites_list_creation(grid, min_c, dim, scale, Top_n, ranking_value)

if len(binding_sites_list) > 0:
    ppdb2 = fromdataframe_topdb(binding_sites_list)
    output_path = ''
    for i in range(len(path) - 4):
        output_path += path[i]
    ppdb2.to_pdb(path = output_path + 'l=' + str(leg) + 'r=' + str(r) + 'spf=' + str(spf) + 'topN=' + str(Top_n) + '.pdb',
             records = ['ATOM'],
             gz = False,
             append_newline = True)
else:
    print("There are no binding sites")