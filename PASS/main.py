import math

import numpy as np
from biopandas.pdb import PandasPdb
from pandas import DataFrame

radii = {
    'N': 1.55,
    'O': 1.52,
    'C': 1.70,
    'F': 1.80,
    'S': 1.80,
    'H': 1.20,
}


def reading_file():
    print('Inserisci il path del file pdb: ')
    path = input()
    ppdb: PandasPdb = PandasPdb().read_pdb(path)
    return ppdb


def hydrogen(ppdb):
    total = len(ppdb.df['ATOM'])
    atoms: DataFrame = ppdb.df['ATOM'][ppdb.df['ATOM']['element_symbol'] == 'H']
    h = len(atoms)
    if h <= 0.2 * total:
        return True
    return False


def delete_hydrogen(x, ppdb):
    if x:
        return ppdb.df['ATOM'][ppdb.df['ATOM']['element_symbol'] != 'H']
    return ppdb


def find_radius(x):
    if x:
        return 1.8
    return 1.5


def find_bc_threshold(x):
    if x:
        return 55
    return 75


def distance(x1, x2, y1, y2, z1, z2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


def initial_layer(matrix, r):
    l = []
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            for k in range(j + 1, len(matrix)):
                R_i = [matrix[i][0], matrix[i][1], matrix[i][2]]
                R_j = [matrix[j][0], matrix[j][1], matrix[j][2]]
                R_k = [matrix[k][0], matrix[k][1], matrix[k][2]]
                o_i = radii[matrix[i][3]]
                o_j = radii[matrix[j][3]]
                o_k = radii[matrix[k][3]]
                if distance(R_i[0], R_j[0], R_i[1], R_j[1], R_i[2], R_j[2]) > o_i + o_j + 2 * r:
                    continue
                if distance(R_i[0], R_k[0], R_i[1], R_k[1], R_i[2], R_k[2]) > o_i + o_k + 2 * r:
                    continue
                if distance(R_j[0], R_k[0], R_j[1], R_k[1], R_j[2], R_k[2]) > o_j + o_k + 2 * r:
                    continue

                T_ij = []
                T_jk = []
                T_ik = []
                for t in range(3):
                    T_ij[t] = 1 / 2 * (R_i[t] + R_j[t]) + ((o_i + r) ** 2 - (o_j + r) ** 2) * (R_j[t] - R_i[t]) / (
                                2 * (distance(R_i[0], R_j[0], R_i[1], R_j[1], R_i[2], R_j[2])) ** 2)
                    T_jk[t] = 1 / 2 * (R_j[t] + R_k[t]) + ((o_j + r) ** 2 - (o_k + r) ** 2) * (R_k[t] - R_j[t]) / (
                                2 * (distance(R_k[0], R_j[0], R_k[1], R_j[1], R_k[2], R_j[2])) ** 2)
                    T_ik[t] = 1 / 2 * (R_i[t] + R_k[t]) + ((o_i + r) ** 2 - (o_k + r) ** 2) * (R_k[t] - R_i[t]) / (
                                2 * (distance(R_k[0], R_i[0], R_k[1], R_i[1], R_k[2], R_i[2])) ** 2)

                R_i = np.array(R_i)
                R_j = np.array(R_j)
                R_k = np.array(R_k)
                T_ik = np.array(T_ik)
                T_ij = np.array(T_ij)
                T_jk = np.array(T_jk)
                U = []
                x = R_j - R_i
                x = x / np.linalg.norm(x)
                y = R_k - R_i
                y = y - (np.dot(y, x) / np.dot(x, x)) * x
                y = y / np.linalg.norm(y)
                U = np.dot(np.array(T_ik - T_ij), np.array(T_ik - R_i)) / np.dot(np.array(T_ik - R_i))

                R_b = R_i + (T_ij - R_i) + U

                h = math.sqrt((o_i + r) ** 2 - distance(R_b[0], R_i[0], R_b[1], R_i[1], R_b[2], R_i[2]))

                z = np.rand(3)
                z = z - (np.dot(z, x) / np.dot(x, x)) * x - (np.dot(z, y) / np.dot(y, y)) * y
                while np.dot(z, z) < 1e-9:
                    z = np.rand(3)
                    z = z - (np.dot(z, x) / np.dot(x, x)) * x - (np.dot(z, y) / np.dot(y, y)) * y

                R_p = R_b + h * z
                l.append(R_p)
                R_p = R_b - h * z
                l.append(R_p)
    return l


def first_filter(l, matrix, r, BC):
    flag = True
    result = []
    for i in range(len(l)):
        flag = True
        count = 0
        for j in range(len(matrix)):
            if distance(l[i][0], matrix[j][0], l[i][1], matrix[j][1], l[i][2], matrix[j][2]) < radii[matrix[j][3]] + r:
                flag = False
                break
            # da chiedere se deve esserci tutta l'atomo o solo il suo centro
            if distance(l[i][0], matrix[j][0], l[i][1], matrix[j][1], l[i][2], matrix[j][2]) <= 8:
                count += 1
        if flag and count >= BC:
            result.append((l[i], count))

    return result

def second_filter(l, r):
    result = []
    for i in range(l):
        flag = True
        for j in range(l):
            if i != j:
                # e se BC fosse uguale?
                if 2 * r + distance(l[i][0], l[j][0], l[i][1], l[j][1], l[i][2], l[j][2]) < 1 and l[i][1] < l[j][1]:
                    flag = False
                    break

        if flag:
            result.append(l[i][0])

    return result








ppdb1 = reading_file()
value = hydrogen(ppdb1)
ppdb1 = delete_hydrogen(value)
r = find_radius(value)
BC = find_bc_threshold(value)
matrix = ppdb1.df['ATOM'][['x_coord', 'y_coord', 'z_coord', 'element_symbol']].to_numpy()
c = initial_layer(matrix, r)
filtered_1 = first_filter(c, matrix, r, BC)
filtered_2 = second_filter(filtered_1, r)



