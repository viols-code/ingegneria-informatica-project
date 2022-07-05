# Computer Science and Engineering Project 2022

## Analysis and implementation of methods for the identification of target sites within proteins to support the discovery of new drugs

**Tutor:** Prof. Gianluca Palermo  
**Area:** Architectures of computer systems  
**Keywords:** Drug Discovery, Virtual Screening, HPC  
**Description:**
The project involves the study and implementation of techniques for the manipulation of small molecules or proteins to
support the discovery of new drugs using computer tools. In particular, the project sees the development and
optimization of techniques known in the literature for the generation of the target pocket from a protein, to support
the virtual screening step using HPC (High-Performance Computing) systems  
– J. Yu, Y. Zhou, I. Tanaka, M. Yao, Roll: A new algorithm for the detection of protein pockets and cavities with a
rolling probe sphere. Bioinformatics, 26(1), 46-52, (2010)  
– Brady, G.P., Stouten, P.F. Fast prediction and visualization of protein binding pockets with PASS. J Comput Aided Mol
Des 14, 383–401 (2000)

## PASS algorithm
If you want to run PASS algorithm, you can use:  
`python main.py input.pdb output.pdb asp.pdb`    
Where:
- 'input.pdb' is the path of the PDB file containing the protein
- 'output.pdb' is the path of the PDB file containing the probe spheres generated by PASS
- 'asp.pdb' is the path of the PDB file containing the ASP generated by PASS

In order to have all the probe spheres founded and so eliminate the last filter, you can use:    
`python main.py input.pdb output.pdb asp.pdb -all`

## POCASA algorithm
If you want to run POCASA algorithm, you can use:  
`python main.py input.pdb output.pdb grid_dimension radius SPF Top_N ranking`  
Where:
- 'input.pdb' is the path of the PDB file containing the protein
- 'output.pdb' is the path of the PDB file containing the probe spheres generated by PASS
- 'grid_dimension' is the dimension of the side of the grid and must have value 0.5 or 1
- 'radius' is the radius of the probe sphere and must have value greater than 1
- 'SPF' is the SPF parameter and must have value between 1 and 26
- 'Top_N' is the Top_n parameter and must have value between 0 and 26
- 'ranking' indicates if you want to sort binding sites groups considering each point as the minumum Manhattan distance
  between him and the board and must be equal to "yes" or "no"

## Tests
In order to test the results of the two methods, two algorithms have been implemented:

## Coverage of the cavities 
If you want to run coverage.py, you can use:    
`python coverage.py result_pocasa.pdb result_pass.pdb`  
Where:  
- 'result_pocasa.pdb' is the path of the result of POCASA
- 'result_pass.pdb' is the path of the result of PASS

## Coverage of the ligand
If you want to run ligand.py, you can use:  
`python ligand.py result.pdb ligand.mol2 radius`  
Where:
- 'result.pdb' is the path of the result of POCASA or PASS
- 'ligand.mol2' is the path of the ligand
- 'radius' is the radius used for the spheres in the result file For PASS we suggest a radius of 1.25, for POCASA a
  radius equals to the dimension of the grid used in the POCASA algorithm
  
In order to run this last algorithm on multiple inputs, you can use the tests.py script.  
You need a directory called 'Proteins' and in this you need a directory for each protein.
The name of the protein PDB file in input must be the name of the directory + '_pocket.pdb' and the name of the ligand PDB file must be the name of the directory + '_ligand.mol2'.  
An example is shown in the figure below:  

<img src="https://github.com/viols-code/ingegneria-informatica-project/blob/master/images/test.png"/>
