# ProjetPython - Protein Ligand Interactions Profiler

Profiler is a class that allows the 3D pharmacophore of a protein-ligand complex to be established. 
From a PDB code or a pathway, it extracts the ligand as well as the residues that constitute the 
pocket. Profiler then builds the pharmacophore profile of the ligand and the pocket. 
Finally, Profiler compares the two profiles to determine the electrostatic interactions that
hat would allow the complex to be stabilised.

## Installation

Using the requirements called "environment_ProjetPython.yml", you can create a conda environement with

    conda env create -f environment_ProjetPython.yml

## Usage

### With pdb and ligand id

        from Pharmacophore3D import Profiler

        p = Profiler()
        p.get_complex(protein='8ef6', ligand='MOI', chain='M')
        p.get_interactions(view=True)

### With pdb and ligand pathway

        from Pharmacophore3D import Profiler

        p = Profiler()
        p.get_complex(protein='pathway/protein.pdb', ligand='pathway/ligand.mol')
        p.get_interactions(view=True)

For general help, questions, suggestions or any other feedback please refer to the [PIA repository](https://github.com/Arthurcarre/ProjetPython).

## Contact

- Mail: [arthur.carre@icloud.com](mailto:arthur.carre@icloud.com)

