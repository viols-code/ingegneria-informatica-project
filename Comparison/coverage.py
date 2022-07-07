import math

from biopandas.pdb import PandasPdb


def reading_file(path):
    """
    Read the PDB file
    """
    ppdb: PandasPdb = PandasPdb().read_pdb(path)
    return ppdb


def matrix_creation(ppdb):
    """
    Create the matrix containing the centers of the atoms with their associated element
    :param ppdb: PandasPdb
    :return: coordinates of centers and other information of atoms in input
    """
    matrix = ppdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord', 'chain_id']].to_numpy()
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
    from sys import argv

    if len(argv) > 2:
        path1 = argv[1]
        path2 = argv[2]
    else:
        print("Insert the path of pocasa output result and the path of pass result")
        exit(1)

    ppdb1 = reading_file(path1)
    pass_results = matrix_creation(ppdb1)

    ppdb2 = reading_file(path2)
    pocasa_results = matrix_creation(ppdb2)

    if not path1.endswith('.pdb'):
        print("The input path must be .pdb")
        exit(1)

    if not path2.endswith('.pdb'):
        print("The input path must be .pdb")
        exit(1)

    pocasa_touched = 0
    pocasa_greater = 0
    pocasa_greater_touched = 0

    for i in range(len(pocasa_results)):
        if pocasa_results[i][3] == 'A':
            pocasa_greater += 1
        point_touched = 0
        for j in range(len(pass_results)):
            if distance(pocasa_results[i][0], pass_results[j][0], pocasa_results[i][1], pass_results[j][1],
                        pocasa_results[i][2], pass_results[j][2]) <= 1.25 + 1e-5:
                point_touched = 1
                break
        if point_touched != 0:
            pocasa_touched += 1
            if pocasa_results[i][3] == 'A':
                pocasa_greater_touched += 1

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
    print('Pocasa greater cavities sites touched by Pass:')
    print((pocasa_greater_touched / pocasa_greater) * 100)
