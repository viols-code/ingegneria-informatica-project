import math
import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from pandas import DataFrame


# inizializzo i raggi dei vari atomi
raggio = {
    'N': 1.6,
    'O': 1.5,
    'C': 1.7,
    'F': 1.8,
    'S': 1.8,
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


def inizializzazione_matrice(ppdb):
    # matrix è un'istanza della classe ndarray
    matrix = ppdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord', 'element_symbol']].to_numpy()

    # per avere tutte le coordinate positive
    for i in range(len(matrix)):
        matrix[i][0] += 50
        matrix[i][1] += 50
        matrix[i][2] += 50

    return matrix


def distanza3D(x1, x2, y1, y2, z1, z2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


def inizializzazione_griglia(matrix):
    # creo una griglia di soli zeri
    grid = np.zeros((200, 200, 200))

    # inserisco gli uni in posizione degli atomi
    # raggio[matrix[i][3]]
    for i in range(len(matrix)):
        intx = int(matrix[i][0])
        inty = int(matrix[i][1])
        intz = int(matrix[i][2])
        for j in range(intx - 2, intx + 3):
            for k in range(inty - 2, inty + 3):
                for t in range(intz - 2, intz + 3):
                    if distanza3D(matrix[i][0], j, matrix[i][1], k, matrix[i][2], t) < raggio[matrix[i][3]]:
                        grid[j][k][t] = 1

    return grid


def creazione_lista_atomi(grid):
    list2 = []
    count = 1
    for i in range(200):
        for j in range(200):
            for k in range(200):
                if grid[i][j][k] == 1:
                    list2.append(
                        ('ATOM', count, '', 'H', '', 'ILE', '', 'A', count, '', '', i - 50, j - 50, k - 50, 1.0, 0,
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













ppdb1 = lettura_file()
ppdb1 = eliminazione_idrogeni(ppdb1)

matrice = inizializzazione_matrice(ppdb1)

griglia = inizializzazione_griglia(matrice)

print('Inserisci il raggio della sfera: ')
r = input()
r = int(r)


lista = creazione_lista_atomi(griglia)

ppdb2 = fromdataframe_topdb(lista)

ppdb2.to_pdb(path='./3eiy_stripped4.pdb',
             records=['ATOM'],
             gz=False,
             append_newline=True)



