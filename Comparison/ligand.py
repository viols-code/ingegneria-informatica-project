import math

from biopandas.mol2 import PandasMol2
from biopandas.pdb import PandasPdb


def reading_file_pdb(path):
    """
    Read the PDB file
    """
    pdb: PandasPdb = PandasPdb().read_pdb(path)
    return pdb


def reading_file_mol2(path):
    """
    Read the PDB file
    """
    mol2: PandasMol2 = PandasMol2().read_mol2(path)
    return mol2


def store_atoms(pdb):
    """
    Return the atoms in the file PDB
    :param pdb: PandasPdb
    """
    return pdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']].to_numpy()


def store_atom_type(mol2):
    """
    Return the atom_type in the file MOL2
    :param mol2: PandasPdb
    """
    return mol2.df[['x', 'y', 'z']]


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


def calculate_percentage(atoms, spheres, r):
    """
    Calculate the percentage of atoms in the ligand covered by the spheres
    :param atoms: the atoms in the ligand
    :param spheres: the spheres obtained with a method
    :param r: the radius
    :return: the percentage of atoms in the ligand covered by the spheres
    """
    coverage = 0
    for index, atom in atoms.iterrows():
        point_touched = 0
        for sphere in spheres:
            if distance(sphere[0], atom['x'], sphere[1], atom['y'], sphere[2], atom['z']) <= r + 1e-5:
                point_touched = 1
                break
        if point_touched != 0:
            coverage += 1

    return coverage / len(atoms)


# ###########
# Entry point
# ###########
if __name__ == '__main__':
    from sys import argv

    # ###########
    # Ligand and output path
    # ###########
    if len(argv) > 3:
        path_result = argv[1]
        path_ligand = argv[2]
        radius = argv[3]
    else:
        print("Insert the path of the output result, the path of the ligand and the radius for the check")
        exit(1)

    if not path_result.endswith('.pdb'):
        print("The result path must be .pdb")
        exit(1)
    if not path_ligand.endswith('.mol2'):
        print("The ligand path must be .mol2")
        exit(1)

    try:
        radius = float(radius)
    except ValueError:
        print("Insert a float as the radius")
        exit(1)

    # ###########
    # Reads the file PDB and MOL2
    # ###########
    try:
        probes = reading_file_pdb(path_result)
    except ValueError or FileNotFoundError:
        print("The input path for the spheres is wrong")
        exit(1)

    try:
        ligand = reading_file_mol2(path_ligand)
    except ValueError or FileNotFoundError:
        print("The input path for the ligand is wrong")
        exit(1)

    # ###########
    # Gets the atoms and spheres coordinates
    # ###########
    ligand = store_atom_type(ligand)
    probes = store_atoms(probes)

    # ###########
    # Calculates the percentage of the coverage
    # ###########
    percentage = calculate_percentage(ligand, probes, radius)
    print("The percentage of the coverage of the ligand is: " + str(percentage * 100) + "%")
