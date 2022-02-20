from biopandas.pdb import PandasPdb
from pandas import DataFrame

# istanziando ppdb della classe PandasPdb
ppdb = PandasPdb()
# leggiamo il file e carichiamo i suoi dati
ppdb.read_pdb('./protein.pdb')

# stampo il contenuto
# print(ppdb.df)

# assegno ad atoms il DataFrame con chiave ATOM

atoms:DataFrame = ppdb.df['ATOM'][ppdb.df['ATOM']['element_symbol'] != 'H']
ppdb.df['ATOM'] = ppdb.df['ATOM'][ppdb.df['ATOM']['element_symbol'] != 'H']
# head di defaul ha n = 5
#print(atoms.head(3217))

atoms = atoms.filter(['x_coord','y_coord','z_coord','element_symbol'])
#print(atoms.head(3217))

#ppdb.to_pdb(path='./3eiy_stripped.pdb',
 #           records=['ATOM'],
  #          gz=False,
   #         append_newline=True)

# print(ppdb.df['ATOM'].head())
# ppdb.read_pdb('./data/3eiy.pdb')
# ppdb:PandasPdb = PandasPdb().fetch_pdb('3eiy')


print(ppdb.df['ATOM'].values)