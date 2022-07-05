import os
import subprocess
import sys

# ###########
# In order for this programs to work, you need a directory called Proteins and insert in this a directory
# for each protein
# ###########
if __name__ == '__main__':
    # Paths
    pocasa_file = 'POCASA/main.py'
    pass_file = 'PASS/main.py'
    output_prefix_pocasa = 'output_pocasa_'
    output_prefix_pass = 'output_pass_'
    ligand = 'Comparison/ligand.py'
    asp = 'asp_'

    # Parameters
    grid_dimension = '1'
    sphere_radius = '4'
    spf = '16'
    top_n = '5'
    ranking = 'yes'

    # Radius for the ligand coverage
    radius_pass = '1.25'
    radius_pocasa = '1'

    for dirs in os.scandir('Proteins'):
        subprocess.call(['python', pass_file, 'Proteins/' + dirs.name + '/' + dirs.name + '_pocket.pdb',
                         'Proteins/' + dirs.name + '/' + output_prefix_pass + dirs.name + '.pdb',
                         'Proteins/' + dirs.name + '/' + asp + dirs.name + '.pdb'],
                        stdout=sys.stdout)

        subprocess.call(['python', pocasa_file, 'Proteins/' + dirs.name + '/' + dirs.name + '_pocket.pdb',
                         'Proteins/' + dirs.name + '/' + output_prefix_pocasa + dirs.name + '.pdb', grid_dimension, sphere_radius, spf, top_n, ranking],
                        stdout=sys.stdout)

        print(dirs.name)
        print("PASS - Ligand coverage: ")
        subprocess.call(['python', ligand, 'Proteins/' + dirs.name + '/' + output_prefix_pass + dirs.name + '.pdb',
                         'Proteins/' + dirs.name + '/' + dirs.name + '_ligand.mol2', radius_pass],
                        stdout=sys.stdout)

        print("POCASA - Ligand coverage: ")
        subprocess.call(['python', ligand, 'Proteins/' + dirs.name + '/' + output_prefix_pocasa + dirs.name + '.pdb',
                         'Proteins/' + dirs.name + '/' + dirs.name + '_ligand.mol2', radius_pocasa],
                        stdout=sys.stdout)