import math
from biopandas.pdb import PandasPdb


def reading_file(path):
    """
    Read the PDB file
    """
    ppdb: PandasPdb = PandasPdb().read_pdb(path)
    return ppdb


def store_atoms(ppdb):
    """
    Return the protein without the hydrogen atoms
    :param pdb: PandasPdb
    """
    ppdb.df['ATOM'] = ppdb.df['ATOM']
    return ppdb


def matrix_creation(ppdb):
    """
    Create the matrix containing the centers of the atoms with their associated element
    :param ppdb: PandasPdb
    :return: coordinates of centers and atomic element of atoms in input
    """
    matrix = ppdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord', 'element_symbol']].to_numpy()
    return matrix


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


# ###########
# Entry point
# ###########
if __name__ == '__main__':
    print('Pocasa results path:')
    path1 = input()
    ppdb1 = reading_file(path1)
    ppdb1 = store_atoms(ppdb1)
    pocasa_results = matrix_creation(ppdb1)

    print('Pass results path:')
    path2 = input()
    ppdb2 = reading_file(path2)
    ppdb2 = store_atoms(ppdb2)
    pass_results = matrix_creation(ppdb2)

    pocasa_touched = 0

    for i in range(len(pocasa_results)):
        point_touched = 0
        for j in range(len(pass_results)):
            if distance(pocasa_results[i][0], pass_results[j][0], pocasa_results[i][1], pass_results[j][1],
                        pocasa_results[i][2], pass_results[j][2]) <= 1.25 + 1e-5:
                point_touched = 1
                break
        if point_touched != 0:
            pocasa_touched += 1

    pass_touched = 0

    for i in range(len(pass_results)):
        point_touched = 0
        for j in range(len(pocasa_results)):
            if distance(pass_results[i][0], pocasa_results[j][0], pass_results[i][1], pocasa_results[j][1],
                        pass_results[i][2], pocasa_results[j][2]) <= 1 + 1e-5:
                point_touched = 1
                break
        if point_touched != 0:
            pass_touched += 1

    pocasa_percentage = pocasa_touched / len(pocasa_results) * 100
    pass_percentage = pass_touched / len(pass_results) * 100
    print('Pocasa binding sites touched by Pass:')
    print(pocasa_percentage)
    print('Pass binding sites touched by Pocasa:')
    print(pass_percentage)


