import math

import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from numba import jit
from pandas import DataFrame

# inizializza i raggi dei vari atomi
raggio_precedente = {
    'N': 1.6,
    'O': 1.5,
    'C': 1.7,
    'F': 1.8,
    'S': 1.8,
    'H': 1,
}

raggio = {
    'N': 1.75,
    'O': 1.55,
    'C': 2.05,
    'F': 2.10,
    'S': 2.10,
    'H': 1,
}

# Lettura del file PDB contenente la proteina utilizzando BioPandas
def lettura_file():
    # chiede all'utente di inserire il path del file pdb
    print('Inserire il path del file pdb: ')
    path = input()
    # legge il file pdb
    ppdb: PandasPdb = PandasPdb().read_pdb(path)
    return ppdb


# Eliminazione di tutti gli idrogeni
def eliminazione_idrogeni(ppdb):
    # elimina gli idrogeni
    ppdb.df['ATOM'] = ppdb.df['ATOM'][ppdb.df['ATOM']['element_symbol'] != 'H']
    return ppdb


# Calcola la cordinata minima tra tutti i punti della matrice,
# che sarà utile per traslare ogni punto così da non avere punti negativi
def trova_minimo(matrice, scala):
    m = 0
    for i in range(len(matrice)):
        m = min(m, min(matrice[i][0], matrice[i][1], matrice[i][2]))
    return m * scala


# Calcola la cordinata massima tra tutti i punti della matrice,
# che sarà utile per definire la dimensione della matrice
def trova_massimo(matrice, scala):
    m = 0
    for i in range(len(matrice)):
        m = max(m, max(matrice[i][0], matrice[i][1], matrice[i][2]))
    return m * scala


# Crea la matrice contente i centri delle proteine in input con associato il loro elemento chimico
def creazione_matrice(ppdb):
    matrix = ppdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord', 'element_symbol']].to_numpy()
    return matrix

# Inizializzazione della matrice che contiene per ogni atomo (tranne l'idrogeno) le coordinate e
# trasla ogni punto con il valore della cordinata minima per avere ogni punto positivo
def inizializzazione_matrice(matrix, m):
    for i in range(len(matrix)):
        matrix[i][0] -= m
        matrix[i][1] -= m
        matrix[i][2] -= m

    return matrix

# Calcola la distanza euclidea tra due punti tridimensionali
@jit(nopython=True)
def distanza3D(x1, x2, y1, y2, z1, z2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

# Crea la griglia e inserisce il valore "1" nei punti che vengono toccati da almeno un atomo
def inizializzazione_griglia(matrix, dimension, scala):
    # Crea una griglia di soli zeri
    grid = np.zeros((dimension*scala + 1, dimension*scala + 1, dimension*scala + 1))

    for i in range(len(matrix)):
        intx = int(matrix[i][0])
        inty = int(matrix[i][1])
        intz = int(matrix[i][2])
        raggio_atomi = 1.8
        if matrix[i][3] in raggio:
            raggio_atomi = raggio[matrix[i][3]]
        else:
            print('Atomo ' + matrix[i][3] + ' non esistente')
        raggio_maggiore = 0
        for key in raggio:
            raggio_maggiore = max(raggio_maggiore, raggio[key])
        raggio_maggiore = int(raggio_maggiore)
        # I punti (j, k, t) sono i punti che potenzialmente possono essere toccati da un atomo.
        for j in range(max(0, (intx - (raggio_maggiore + 2)) * scala), min(dimension + 1, (intx + (raggio_maggiore + 2)) * scala + 1)):
            for k in range(max(0, (inty - (raggio_maggiore + 2)) * scala), min(dimension + 1, (inty + (raggio_maggiore + 2)) * scala + 1)):
                for t in range(max(0, (intz - (raggio_maggiore + 2)) * scala), min(dimension + 1, (intz + (raggio_maggiore + 2)) * scala + 1)):
                    if distanza3D(matrix[i][0], j / scala, matrix[i][1], k / scala, matrix[i][2], t / scala) < raggio_atomi:
                        grid[j][k][t] = 1

    return grid

# Calcola la distanza dal bordo per ogni punto appartente a un binding sites
def distanza_dal_bordo(x, y, z, griglia, dimensione):
    # definisce la distanza minima a "1"
    distanza = 1
    while(True):
        for i in range(max(0, x - distanza), min(dimensione + 1, x + distanza + 1)):
            for j in range(max(0, y - distanza), min(dimensione + 1, y + distanza + 1)):
                for k in range(max(0, z - distanza), min(dimensione + 1, z + distanza + 1)):
                    if griglia[i][j][k] == -1:
                        return distanza
        distanza += 1

# Ordina gli elemneti della lista in ingresso utilizzando l'algoritmo di Counting Sort
def counting_sort(list1, dim, griglia, dimensione):
    counting = np.zeros((dim + 1))
    for i in range(len(list1)):
        distanza = distanza_dal_bordo(int(list1[i][1] * scala), int(list1[i][2] * scala), int(list1[i][3] * scala), griglia, dimensione)
        counting[list1[i][0]] += 1
    copy_counting = []
    for i in range(2, dim + 1):
        copy_counting.append(counting[i])
    copy_counting.sort()
    copy_counting.reverse()
    used = np.zeros((dim + 1))
    list2 = []
    for i in range(len(copy_counting)):
        current_value = 0
        for j in range(2, dim + 1):
            if(counting[j] == copy_counting[i] and used[j] == 0):
                used[j] = 1
                current_value = j
                break
        for j in range(len(list1)):
            if list1[j][0] == current_value:
                list2.append(list1[j])
    return list2


# Creazione del file PDB contenente i possibili binding sites
def creazione_lista_atomi(grid, m, dimensione, scala, Top_n):
    list1 = []
    lettere ="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    dim = 2
    for i in range(dimensione + 1):
        for j in range(dimensione + 1):
            for k in range(dimensione + 1):
                if grid[i][j][k] >= 2:
                    list1.append((int(grid[i][j][k]), i / scala + m, j / scala + m, k / scala + m))
                    dim = max(dim, int(grid[i][j][k]))
    # La lista di gruppi di binding sites viene ordinata in maniera decrescente in base alle loro dimensioni
    list1 = counting_sort(list1, dim, grid, dimensione)
    list2 = []
    count = 1
    current_letter = 0
    # Ogni punto appartenente al binding site viene inserito in list2
    for i in range(len(list1)):
        if(i!=0 and list1[i][0] != list1[i-1][0]):
            current_letter += 1
        if current_letter == Top_n:
            break
        list2.append(('ATOM', count, '', 'H', '', 'ILE', '', lettere[current_letter], count, '', '', float(list1[i][1]) , float(list1[i][2]), float(list1[i][3]),
                    1.0 , 0,'', '', 'H', 0, count))
        count += 1
    return list2


# La lista contenente tutti i punti del binding site viene utilizzata per creare il file PDB
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

# Se tutti i punti attorno (ovvero, che rientrano nella sfera di raggio r) non fanno parte della
# proteina, allora è parte del bordo
@jit(nopython=True)
def parte_del_bordo(x, y, z, r, griglia, dimensione, scala):
    for i in range(max(0, (x - r * scala), min(dimensione + 1, x + r * scala + 1))):
        for j in range(max(0, y - r * scala), min(dimensione + 1, y + r * scala + 1)):
            for k in range(max(0, z - r * scala), min(dimensione + 1, z + r * scala + 1)):
                if griglia[i][j][k] == 1 and distanza3D(x / scala, i / scala, y / scala, j / scala, z / scala, k / scala) < r:
                    return 0
    return 1


# Per ogni punto che è parte del bordo tutti i punti contenuti nella sfera vengono posti a "-1"
@jit(nopython=True)
def crea_bordo(griglia, r, dimensione, scala, m, M):
    for i in range(dimensione + 1):
        for j in range(dimensione + 1):
            for k in range(dimensione + 1):
                if parte_del_bordo(i, j, k, r, griglia, dimensione, scala):
                    for t in range(max(0, i - r * scala), min(dimensione + 1, i + r * scala + 1)):
                        for v in range(max(0, j - r * scala), min(dimensione + 1, j + r * scala + 1)):
                            for u in range(max(0, k - r * scala), min(dimensione + 1, k + r * scala + 1)):
                                if distanza3D(i / scala, t / scala, j /scala, v / scala, k / scala, u /scala) < r:
                                    griglia[t][v][u] = -1
    return griglia


# Per ogni punto che potrebbe costituire un binding site, viene contato il parametro countspf
# che conta il numero di punti appartenenti al binding site attorno al punto considerato e
# se il valore appena calcolato é inferiore al valore di soglia lo elimina dai punti apparteneti ai binding sites
@jit(nopython=True)
def rimoziozione_noise_points(griglia, dimensione, scala, spf):
    griglia2 = np.zeros((dimensione + 1, dimensione + 1, dimensione + 1))
    for i in range(dimensione + 1):
        for j in range(dimensione + 1):
            for k in range(dimensione + 1):
                griglia2[i][j][k] = griglia[i][j][k]
    for i in range(1, dimensione):
        for j in range(1, dimensione):
            for k in range(1, dimensione):
                if griglia2[i][j][k] != 0:
                    continue
                countspf = 0
                for c1 in range(i - 1, i + 2):
                    for c2 in range(j - 1, j + 2):
                        for c3 in range(k - 1, k + 2):
                            if c1 == i and c2 == j and c3 == k:
                                continue
                            if griglia2[c1][c2][c3] == 0:
                                countspf += 1
                if countspf <= spf:
                    griglia[i][j][k] = -2
    return griglia


# Associa ad ogni gruppo di binding sites un valore che poi diventerà una lettera
# Serve per identificare i diversi gruppi
@jit(nopython=True)
def bfs(griglia, dimensione, scala):
    queue = []
    counter = 1
    for i in range(dimensione + 1):
        for j in range(dimensione + 1):
            for k in range(dimensione + 1):
                if(griglia[i][j][k] == 0):
                    counter += 1
                    griglia[i][j][k] = counter
                    queue.append((i, j, k))
                while(len(queue) != 0):
                    h = queue.pop()
                    x = h[0]
                    y = h[1]
                    z = h[2]
                    for c1 in range(max(0, x - 1), min(dimensione + 1, x + 2)):
                        for c2 in range(max(0, y - 1), min(dimensione + 1, y + 2)):
                            for c3 in range(max(0, z-1), min(dimensione + 1, z + 2)):
                                if griglia[c1][c2][c3] == 0:
                                    griglia[c1][c2][c3] = counter
                                    queue.append((c1, c2, c3))
    return griglia

print('Inserire il lato della griglia: ')
lato = input()
lato = float(lato)

while(lato != 0.5 and lato != 1):
    print('Inserire il lato della griglia: ')
    lato = input()
    lato = float(lato)

scala = 1
if lato == 0.5:
    scala = 2

ppdb1 = lettura_file()
ppdb1 = eliminazione_idrogeni(ppdb1)

matrice = creazione_matrice(ppdb1)
max_c = trova_massimo(matrice, scala)
max_c = int(max_c) + 8 * scala + 1
min_c = trova_minimo(matrice, scala)
min_c = int(min_c) - 8 * scala
matrice = inizializzazione_matrice(matrice, min_c)
griglia = inizializzazione_griglia(matrice, max_c - min_c, scala)


print('Inserire il raggio della sfera: ')
r = input()
r = int(r)

print('Inserire il valore di soglia SPF (valore da 0 a 27): ')
spf = input()
spf = int(spf)
while spf < 0 or spf > 27:
    print('Inserire il valore di soglia SPF (valore da 0 a 27): ')
    spf = input()
    spf = int(spf)

print('Inserire quanti binding sites visualizzare (valore da 1 a 26): ')
Top_n = input()
Top_n = int(Top_n)
while Top_n < 1 or Top_n > 26:
    print('Inserire quanti binding sites visualizzare (valore da 1 a 26): ')
    Top_n = input()
    Top_n = int(Top_n)

griglia = crea_bordo(griglia, r, max_c - min_c, scala, min_c, max_c)

griglia = rimoziozione_noise_points(griglia, max_c - min_c, scala, spf)

griglia = bfs(griglia, max_c - min_c, scala)

lista = creazione_lista_atomi(griglia, min_c, max_c - min_c, scala, Top_n)

if len(lista) > 0:

    ppdb2 = fromdataframe_topdb(lista)

    ppdb2.to_pdb(path='output4.pdb',
             records=['ATOM'],
             gz=False,
             append_newline=True)

else:

    print("Non ci sono binding sites")