import math

import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from numba import jit
from pandas import DataFrame

# inizializzo i raggi dei vari atomi
raggio = {
    'N': 1.6,
    'O': 1.5,
    'C': 1.7,
    'F': 1.8,
    'S': 1.8,
    'H': 1,
}


def lettura_file():
    # chiedo all'utente di inserire il path del file pdb
    print('Inserisci il path del file pdb: ')
    path = input()
    # leggo il file pdb
    ppdb: PandasPdb = PandasPdb().read_pdb(path)
    return ppdb


def eliminazione_idrogeni(ppdb):
    # elimino gli idrogeni
    ppdb.df['ATOM'] = ppdb.df['ATOM'][ppdb.df['ATOM']['element_symbol'] != 'H']
    return ppdb


def trova_minimo(matrice):
    m = 0
    for i in range(len(matrice)):
        m = min(m, min(matrice[i][0], matrice[i][2], matrice[i][2]))
    return m


def trova_massimo(matrice):
    m = 0
    for i in range(len(matrice)):
        m = max(m, max(matrice[i][0], matrice[i][2], matrice[i][2]))
    return m


def creazione_matrice(ppdb):
    # matrix Ã¨ un'istanza della classe ndarray
    matrix = ppdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord', 'element_symbol']].to_numpy()
    return matrix


def inizializzazione_matrice(matrix, m):
    # per avere tutte le coordinate positive
    for i in range(len(matrix)):
        matrix[i][0] += m
        matrix[i][1] += m
        matrix[i][2] += m

    return matrix


@jit(nopython=True)
def distanza3D(x1, x2, y1, y2, z1, z2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


def inizializzazione_griglia(matrix, dimension, scala):
    # creo una griglia di soli zeri
    grid = np.zeros((dimension * scala + 1, dimension * scala + 1, dimension * scala + 1))

    # inserisco gli uni in posizione degli atomi
    # raggio[matrix[i][3]]
    for i in range(len(matrix)):
        intx = int(matrix[i][0])
        inty = int(matrix[i][1])
        intz = int(matrix[i][2])
        raggio_atomi = 1.8
        if raggio[matrix[i][3]] in raggio:
            raggio_atomi = raggio[matrix[i][3]]
        for j in range(intx - 2 * scala, intx + 2 * scala + 1):
            for k in range(inty - 2 * scala, inty + 2 * scala + 1):
                for t in range(intz - 2 * scala, intz + 2 * scala + 1):
                    if distanza3D(matrix[i][0], j, matrix[i][1], k, matrix[i][2], t) < raggio_atomi * scala:
                        grid[j][k][t] = 1

    return grid


@jit(nopython=True)
def creazione_lista_atomi(grid, m, dimensione, scala):
    list2 = []
    count = 1
    for i in range(dimensione * scala + 1):
        for j in range(dimensione * scala + 1):
            for k in range(dimensione * scala + 1):
                if grid[i][j][k] == 0:
                    list2.append(
                        ('ATOM', count, '', 'H', '', 'ILE', '', 'A', count, '', '', (i - m) / scala, (j - m) / scala, (k - m) / scala, 1.0, 0,
                         '', '', 'H', 0, count))
                    count += 1

    return list2


def fromdataframe_topdb(list1):
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


@jit(nopython=True)
def parte_del_bordo(x, y, z, r, griglia, dimensione, scala):
    for i in range(max(0, ((x - r) * scala), min(dimensione * scala + 1, (x + r) * scala + 1))):
        for j in range(max(0, (y - r) * scala), min(dimensione * scala + 1, (y + r) * scala + 1)):
            for k in range(max(0, (z - r) * scala), min(dimensione * scala + 1, (z + r) * scala + 1)):
                if griglia[i][j][k] == 1 and distanza3D(x, i, y, j, z, k) < r * scala:
                    return 0
    return 1


@jit(nopython=True)
def crea_bordo(griglia, r, dimensione, scala):
    for i in range(dimensione * scala + 1):
        for j in range(dimensione * scala + 1):
            for k in range(dimensione * scala + 1):
                if parte_del_bordo(i, j, k, r, griglia, dimensione, scala):
                    for t in range(max(0, i - r * scala), min(dimensione * scala + 1, i + r * scala + 1)):
                        for v in range(max(0, j - r * scala), min(dimensione * scala + 1, j + r * scala + 1)):
                            for u in range(max(0, k - r * scala), min(dimensione * scala + 1, k + r * scala + 1)):
                                if distanza3D(i, t, j, v, k, u) < r * scala:
                                    griglia[t][v][u] = -1
    return griglia


@jit(nopython=True)
def rimoziozione_noise_points(griglia, dimensione, scala):
    griglia2 = np.zeros((dimensione * scala + 1, dimensione * scala + 1, dimensione * scala + 1))
    for i in range(dimensione * scala + 1):
        for j in range(dimensione * scala + 1):
            for k in range(dimensione * scala + 1):
                griglia2[i][j][k] = griglia[i][j][k]
    for i in range(1, dimensione * scala):
        for j in range(1, dimensione * scala):
            for k in range(1, dimensione * scala):
                if griglia2[i][j][k] != 0:
                    continue
                countspf = 0
                countpdf = 0
                for c1 in range(i - 1, i + 2):
                    for c2 in range(j - 1, j + 2):
                        for c3 in range(k - 1, k + 2):
                            if c1 == i and c2 == j and c3 == k:
                                continue
                            if griglia2[c1][c2][c3] == 0:
                                countspf += 1
                            if griglia2[c1][c2][c3] == 1:
                                countpdf += 1
                if countspf <= 16:  # and countpdf <= 18
                    griglia[i][j][k] = -3
    return griglia


print('Inserisci il lato della griglia: ')
lato = input()
lato = int(lato)

while(lato != 0.5 and lato != 1):
    print('Inserisci il lato della griglia: ')
    lato = input()
    lato = int(lato)

scala = 1
if lato == 0.5:
    scala = 2

ppdb1 = lettura_file()
ppdb1 = eliminazione_idrogeni(ppdb1)

matrice = creazione_matrice(ppdb1)
max_c = trova_massimo(matrice)
max_c = int(max_c) + 17
min_c = trova_minimo(matrice)
min_c = int(min_c) - 16
matrice = inizializzazione_matrice(matrice, min_c)
griglia = inizializzazione_griglia(matrice, max_c - min_c, scala)

print('Inserisci il raggio della sfera: ')
r = input()
r = int(r)

griglia = crea_bordo(griglia, r, max_c - min_c, scala)

griglia = rimoziozione_noise_points(griglia, max_c - min_c, scala)

lista = creazione_lista_atomi(griglia, min_c, max_c - min_c, scala)

ppdb2 = fromdataframe_topdb(lista)

ppdb2.to_pdb(path='./1a0t_binding_sites_radius_2.pdb',
             records=['ATOM'],
             gz=False,
             append_newline=True)
