from biopandas.pdb import PandasPdb
import numpy as np
import math
import pandas as pd
from pandas import DataFrame

#inizializzo i raggi dei vari atomi
raggio = {
    'N' : 1.6,
    'O' : 1.5,
    'C' : 1.7,
    'F' : 1.8,
    'S' : 1.8,
}

def letturaFile():
    # chiedo all'utente di inserire il path del file pdb
    print('Inserisci il path del file pdb')
    path = input()
    # leggo il file pdb
    ppdb:PandasPdb = PandasPdb().read_pdb(path)
    return ppdb

def eliminazioneIdrogeni(ppdb):
    #elimino gli idrogeni
    ppdb.df['ATOM'] = ppdb.df['ATOM'][ppdb.df['ATOM']['element_symbol'] != 'H']
    return ppdb

def inizializzazioneMatrice():
    # matrix è un'istanza della classe ndarray
    matrix = ppdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord', 'element_symbol']].to_numpy()

    # per avere tutte le coordinate positive
    for i in range(len(matrix)):
        matrix[i][0] += 50
        matrix[i][1] += 50
        matrix[i][2] += 50

    return matrix

def inizializzazioneGriglia():
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
                    if math.sqrt((matrix[i][0] - j) ** 2 + (matrix[i][1] - k) ** 2 + (matrix[i][2] - t) ** 2) < raggio[
                        matrix[i][3]]:
                        grid[j][k][t] = 1

    return grid

def fromDataFrametoPdb(list):
    pp: DataFrame = pd.DataFrame(list,
                                 columns=['record_name', 'atom_number', 'blank_1', 'atom_name', 'alt_loc',
                                          'residue_name', 'blank_2',
                                          'chain_id', 'residue_number', 'insertion', 'blank_3', 'x_coord', 'y_coord',
                                          'z_coord',
                                          'occupancy', 'b_factor', 'blank_4', 'segment_id', 'element_symbol', 'charge',
                                          'line_idx'])

    ppdb = PandasPdb()
    ppdb.df['ATOM'] = pp

    return ppdb

def creazioneListaAtomi():
    list = []
    count = 1
    for i in range(200):
        for j in range(200):
            for k in range(200):
                if (grid[i][j][k] == 1):
                    list.append(
                        ('ATOM', count, '', 'H', '', 'ILE', '', 'A', count, '', '', i - 50, j - 50, k - 50, 1.0, 0,
                         '', '', 'H', 0, count))
                    count += 1

    return list



ppdb = letturaFile()
ppdb = eliminazioneIdrogeni(ppdb)

matrix = inizializzazioneMatrice()

grid = inizializzazioneGriglia()

list = creazioneListaAtomi()

pppdb = fromDataFrametoPdb(list)

pppdb.to_pdb(path='./3eiy_stripped4.pdb',
            records=['ATOM'],
            gz=False,
            append_newline=True)
