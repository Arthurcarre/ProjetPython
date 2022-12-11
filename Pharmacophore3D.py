# 
import os
import requests
import pymol2
import numpy as np
import pandas as pd
import subprocess
import warnings
from biopandas.pdb import PandasPdb
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from bs4 import BeautifulSoup

warnings.filterwarnings('ignore')

"""
The following three functions are from the TeachOpenCADD talktorial T008 module.
They allow, from a PDB code, to restore the SMILES code of the ligands that the PDB file contains.
This is useful here to reassign the order of double and triple bonds.
"""

def get_pdb_ligands(pdb_id):
    """
    RCSB has not provided a new endpoint for ligand information yet. As a
    workaround we are obtaining extra information from ligand-expo.rcsb.org,
    using HTML parsing. Check Talktorial T011 for more info on this technique!
    """
    pdb_info = _fetch_pdb_nonpolymer_info(pdb_id)
    ligand_expo_ids = [
        nonpolymer_entities["pdbx_entity_nonpoly"]["comp_id"]
        for nonpolymer_entities in pdb_info["data"]["entry"]["nonpolymer_entities"]
    ]

    ligands = {}
    for ligand_expo_id in ligand_expo_ids:
        ligand_expo_info = _fetch_ligand_expo_info(ligand_expo_id)
        ligands[ligand_expo_id] = ligand_expo_info

    return ligands


def _fetch_pdb_nonpolymer_info(pdb_id):
    """
    Fetch nonpolymer data from rcsb.org.
    Thanks @BJWiley233 and Rachel Green for this GraphQL solution.
    """
    query = (
        """{
          entry(entry_id: "%s") {
            nonpolymer_entities {
              pdbx_entity_nonpoly {
                comp_id
                name
                rcsb_prd_id
              }
            }
          }
        }"""
        % pdb_id
    )

    query_url = f"https://data.rcsb.org/graphql?query={query}"
    response = requests.get(query_url)
    response.raise_for_status()
    info = response.json()
    return info


def _fetch_ligand_expo_info(ligand_expo_id):
    """
    Fetch ligand data from ligand-expo.rcsb.org.
    """
    r = requests.get(f"http://ligand-expo.rcsb.org/reports/{ligand_expo_id[0]}/{ligand_expo_id}/")
    r.raise_for_status()
    html = BeautifulSoup(r.text)
    info = {}
    for table in html.find_all("table"):
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) != 2:
                continue
            key, value = cells
            if key.string and key.string.strip():
                info[key.string.strip()] = "".join(value.find_all(string=True))

    # Postprocess some known values
    info["Molecular weight"] = float(info["Molecular weight"].split()[0])
    info["Formal charge"] = int(info["Formal charge"])
    info["Atom count"] = int(info["Atom count"])
    info["Chiral atom count"] = int(info["Chiral atom count"])
    info["Bond count"] = int(info["Bond count"])
    info["Aromatic bond count"] = int(info["Aromatic bond count"])
    return info


def get_distance_between_2_points(point1, point2):
    """
    Calculating the Euclidean distance between two points in a three-dimensional space
    Parameters :
        Input :
            - point1 : tuple of three floats corresponding to the coordinates of the point 1
            - point2 : tuple of three floats corresponding to the coordinates of the point 2
        Output :
            - distance : Result of the calculation of the Euclidean distance in the format np.float
    """
    
    distance =  np.sqrt([  (point1[0] - point2[0])**2
                         + (point1[1] - point2[1])**2
                         + (point1[2] - point2[2])**2
                           ])
    
    return distance

def get_angle(point1, point2, point3):
    """
    Calculating the angle between three points in a three-dimensional space
    Parameters :
        Input :
            - point1 : tuple of three floats corresponding to the coordinates of the point 
            which lies between the other two points
            - point2 : tuple of three floats corresponding to the coordinates of the point 2
            - point3 : tuple of three floats corresponding to the coordinates of the point 2
        Output :
            - distance : Result of the calculation of the Euclidean distance in the format np.float
    """ 
    P12 = np.sqrt((point1.x_coord - point2.x_coord)**2
                 +(point1.y_coord - point2.y_coord)**2
                 +(point1.z_coord - point2.z_coord)**2)
    P13 = np.sqrt((point1.x_coord - point3.x_coord)**2
                 +(point1.y_coord - point3.y_coord)**2
                 +(point1.z_coord - point3.z_coord)**2)
    P23 = np.sqrt((point3.x_coord - point2.x_coord)**2
                 +(point3.y_coord - point2.y_coord)**2
                 +(point3.z_coord - point2.z_coord)**2)
    
    return np.rad2deg(np.arccos((P12**2 + P13**2 - P23**2) / (2 * P12 * P13)))

def protein_initialization(protein):
    """
    Set the index of the pandas array corresponding to the protein. Each line corresponds to an atom.
    Each atom has, as an index, the identifier of the residue (example: TYR)
    to which it belongs, then the number of this residue (example: 18), then the number of the protein chain
    to which the residue belongs (example: A), followed by an underscore '_',
    followed by the name of the atom (example: CA). Example: TYR18A_CA
    
    Parameters :
        Input :
            - protein : dataframe of the protein in pandas format from the Biopandas module.
    """
    
    protein['residue'] = 0
    protein['indice'] = 0
    protein['res_id'] = 0
    for i in range(len(protein)):
        protein['residue'][i] = str(protein['alt_loc'][i] +
                                      protein['residue_name'][i] +
                                      str(protein['residue_number'][i]) +
                                      protein['chain_id'][i])
        protein['indice'][i] = str(protein['alt_loc'][i] +
                                     protein['residue_name'][i] +
                                     str(protein['residue_number'][i]) +
                                     protein['chain_id'][i] + "_" +
                                     protein['atom_name'][i])
        protein['res_id'][i] = str(str(protein['residue_number'][i]) +
                                     protein['chain_id'][i])
    protein.set_index('indice', inplace = True)

def get_ligand_with_pymol(pathway_molecule, ligand):
    """
    If the protein comes from a local PDB file, this function recreates the ligand by reassigning
    the order of the double and triple bonds, as well as adding the hydrogens, through PyMOL.
    Parameters :
        Input :
            - pathway_molecule : pathway of the molecule isolated and written with the Biopandas module
            - ligand : dataframe of the ligand in pandas format from the Biopandas module.
        Output :
            - ligand in Mol File
    """
    p1 = pymol2.PyMOL()
    p1.start()
    p1.cmd.load(pathway_molecule, 'mol')
    p1.cmd.h_add()
    p1.cmd.save('pymol_ligand.mol')
    p1.stop()

    return Chem.MolFromMolFile('pymol_ligand.mol', removeHs=False)


def get_ligand_with_smiles(pathway_molecule, smiles):
    """
    If the protein comes from a PDB ID, this function recreates the ligand by reassigning
    the order of the double and triple bonds, as well as adding the hydrogens, through the smiles code.
    Parameters :
        Input :
            - pathway_molecule : pathway of the molecule isolated and written with the Biopandas module
            - smiles : ligand smiles code
        Output :
            - ligand in Mol File
    """ 
    mol = Chem.MolFromPDBFile(pathway_molecule)
    template = Chem.MolFromSmiles(smiles)
    
    return Chem.AddHs(AllChem.AssignBondOrdersFromTemplate(template, mol),
                      addCoords=True,
                      addResidueInfo=True)

def ligand_initialization(ligand):
    """
    Set the index of the pandas array corresponding to the ligand. Each line corresponds to an atom.
    Each atom has, as an index, the identifier of the residue (example: QNB)
    to which it belongs, then the number of this residue (example: 108), then the number of the protein chain
    to which the residue belongs (example: A), followed by an underscore '_',
    followed by the name of the atom (example: C14). Example: QNB108A_C14
    Parameters :
        Input :
            - ligand : dataframe of the ligand in pandas format from the Biopandas module.
    """ 
    
    ligand['indice'] = 'na'
    for i in ligand.index:
        ligand['indice'][i] = str(ligand['alt_loc'][i] +
                                       ligand['residue_name'][i] +
                                       str(ligand['residue_number'][i]) +
                                       ligand['chain_id'][i] + "_" +
                                       ligand['atom_name'][i])
    ligand.set_index('indice', inplace = True)

def get_pocket(protein, ligand, size_pocket):
    """
    From the selected ligand within the previously selected protein chain, the pocket 
    is defined as all residues that are within SIZE_POCKET Å of each atom (except hydrogen) of the ligand. 
    hydrogen) of the ligand.
    Parameters :
        Input :
            - ligand : dataframe of the ligand in pandas format from the Biopandas module.
            - protein : dataframe of the protein in pandas format from the Biopandas module.
            - size_pocket : float which defines the maximum distance between a ligand atom and a protein atom
            of the protein, for which the corresponding residue is considered as constituting the pocket.
        Output :
            - pocket : dataframe of the pocket in pandas format from the Biopandas module.
            - pocket_structure : pocket in rdKit mol file
    """    
    iteration = 0
    indice_atom_pocket = []
    for atom_protein in protein.index :
        if protein.loc[atom_protein, "element_symbol"] != 'H' :
            for atom_ligand in ligand.index :
                if ligand.loc[atom_ligand, "element_symbol"] != 'H' :
                    if (protein.loc[atom_protein, "x_coord"] <
                        (ligand.loc[atom_ligand, "x_coord"]) + size_pocket) and (
                    (protein.loc[atom_protein, "x_coord"] >
                     (ligand.loc[atom_ligand, "x_coord"] - size_pocket))) :
                        if (protein.loc[atom_protein, "y_coord"] <
                            (ligand.loc[atom_ligand, "y_coord"]) + size_pocket) and (
                    (protein.loc[atom_protein, "y_coord"] >
                     (ligand.loc[atom_ligand, "y_coord"] - size_pocket))) :
                            if (protein.loc[atom_protein, "z_coord"] <
                                (ligand.loc[atom_ligand, "z_coord"]) + size_pocket) and (
                    (protein.loc[atom_protein, "z_coord"] >
                     (ligand.loc[atom_ligand, "z_coord"] - size_pocket))) :
                                
                                iteration += 1
                                if get_distance_between_2_points(
                                    (ligand.loc[atom_ligand, "x_coord"],
                                     ligand.loc[atom_ligand, "y_coord"],
                                     ligand.loc[atom_ligand, "z_coord"]),
                                    (protein.loc[atom_protein, "x_coord"],
                                     protein.loc[atom_protein, "y_coord"],
                                     protein.loc[atom_protein, "z_coord"])
                                ) < size_pocket :
                                    
                                    indice_atom_pocket.append(atom_protein)
    #print(iteration)                                
    atom_pocket = protein.loc[indice_atom_pocket] 
    indice_res_pocket = []
    for atom in protein.index :
        if protein.loc[atom, "res_id"] in set(atom_pocket["res_id"]) :
            indice_res_pocket.append(atom)
    pocket = PandasPdb().read_pdb('complex_H.pdb')
    os.remove('complex_H.pdb')
    pocket.df["ATOM"] = protein.loc[indice_res_pocket]
    del pocket.df['HETATM']
    pocket.to_pdb(path='pocket.pdb', 
                records=None, 
                gz=False, 
                append_newline=True)
    print(f"\nThe residues which constitute the pocket are : {set(pocket.df['ATOM']['residue'])}")
    
    return protein.loc[indice_res_pocket], Chem.MolFromPDBFile('pocket.pdb', removeHs=False) 
  
def caracterisation(molecule, molecule_structure, type_molecule=False):
    """
    This function takes a molecule as input (ligand or pocket for example) and determines for each atom its
    atom its hybridisation as well as its neighbors. It then establishes the pharmacophore profile of the
    molecule and stores all the information in a dataframe.
    Parameters :
        Input :
            - molecule : dataframe of the molecule in pandas format from the Biopandas module.
            - molecule_structure : molecule in Mol File
            - type_molecule : If it is the pocket, it converts the nitrogen of the isolated backbone 
            (i.e. the nitrogen of the pocket residues that are no longer linked to the residues that are not part of the pocket. 
            to the residues that are not part of the pocket) into SP2 hybridisation.
        Output :
            - molecule : dataframe of the molecule in pandas format from the Biopandas module including
            the hybridisation of each atom as well as its neighbors
            - profile : dataframe including each pharmacophore featuring per row. In columns, there are the 
            family, the type and the coordonates of each pharmacophore featuring as well as the identifier of
            each atom involved in the pharmacophore featuring.
    """           
    index = molecule.index
    atomic_number = pd.Series([atom.GetIdx() for atom in molecule_structure.GetAtoms()], index = index)
    symbol = pd.Series([atom.GetSymbol() for atom in molecule_structure.GetAtoms()], index = index)
    neighbors_rdkit = pd.Series([
        atom.GetNeighbors() for atom in molecule_structure.GetAtoms()], index = index)
    hybridation = pd.Series([str(atom.GetHybridization()) for atom in molecule_structure.GetAtoms()],
                            index = index)
    
    rdkit_df = pd.DataFrame({'Atom Symbol': symbol,
                           'Atom Number': atomic_number,
                           'Hybridation': hybridation,
                           'Neighbors RDKit': neighbors_rdkit})
    
    molecule = pd.concat([molecule, rdkit_df], axis=1, verify_integrity=True)
    neighbors = pd.Series([tuple([molecule[molecule['Atom Number'] == neighbor.GetIdx()].index[0]
         for neighbor in molecule.loc[index, 'Neighbors RDKit']]) for index in molecule.index],
                      index = molecule.index)
    molecule['Neighbors'] = neighbors
    
    fdefName = 'BaseFeatures.fdef'
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    feats = factory.GetFeaturesForMol(molecule_structure)

    features_family = [i.GetFamily() for i in feats]
    features_type = [i.GetType() for i in feats]
    feat_x_coor = [i.GetPos()[0] for i in feats]
    feat_y_coor = [i.GetPos()[1] for i in feats]
    feat_z_coor = [i.GetPos()[2] for i in feats]
    atom_index = [[molecule.index[feat.GetAtomIds()[i]]
                    for i in range(len(feat.GetAtomIds()))]
                     for feat in feats]
    
    zipped = list(zip(features_family, features_type, feat_x_coor, feat_y_coor, feat_z_coor, atom_index))
    profile = pd.DataFrame(
            zipped, columns=['Features_family','Features_type','x_coord','y_coord','z_coord', 'atom_Index'])
    
    if type_molecule == 'Pocket' :
        for index_mol in molecule.index :
            if molecule.loc[index_mol, 'atom_name'] == 'N':
                molecule.loc[index_mol, 'Hybridation'] = 'SP2'     

    return molecule, profile

def get_hydrophobic_interactions(ligand,
                                 ligand_profile,
                                 pocket,
                                 pocket_profile,
                                 hydrophobic_bond_distance):
    """
    Computation of hydrophobic interactions between the ligand and the pocket.
    Parameters :
        Input :
            - ligand : dataframe of the ligand in pandas format from the Biopandas module.
            - ligand_profile : ligand in Mol File
            - pocket : dataframe of the pocket in pandas format from the Biopandas module.
            - pocket_profile : pocket in Mol File
            - hydrophobic_bond_distance : float which defines the maximum distance between two
            hydrophobic atoms between the ligand and the pocket, for which we consider the
            hydrophobic interaction as valid.
        Output :
            - results_hydrophobic_interactions : hydrophobic interaction dataframe including the residues,
            residue numbers, distance, ligand and protein atom numbers, ligand and protein atom symbols,
            which are involved in each interaction.
    """           
    residues = []
    aa = []
    distances = []
    atom_ligand_id = []
    atom_ligand_symbol = []
    protein_atom_id = []
    protein_atom_symbol = []
    feat_lig_index = []
    feat_pock_index = []

    for feat_lig in ligand_profile.index :
        if ligand_profile.loc[feat_lig, 'Features_family'] == 'Hydrophobe'  :
            for feat_pock in pocket_profile.index :
                if pocket_profile.loc[feat_pock, 'Features_family'] == 'Hydrophobe'   :
                    distance = get_distance_between_2_points((ligand_profile.loc[feat_lig, "x_coord"],
                                                              ligand_profile.loc[feat_lig, "y_coord"],
                                                              ligand_profile.loc[feat_lig, "z_coord"]),
                                                             (pocket_profile.loc[feat_pock, "x_coord"], 
                                                              pocket_profile.loc[feat_pock, "y_coord"],
                                                              pocket_profile.loc[feat_pock, "z_coord"]))
                    if distance < hydrophobic_bond_distance :

                        distances.append(round(float(distance), 2))
                        feat_lig_index.append(feat_lig)
                        feat_pock_index.append(feat_pock)

                        protein_atom_id.append(str(set([pocket.loc[
                                pocket_profile.loc[feat_pock, "atom_Index"][i],
                                'atom_number'] for i in range(len(pocket_profile.loc[
                                    feat_pock, "atom_Index"]))]))[1:-1])

                        atom_ligand_id.append(str([ligand.loc[
                                ligand_profile.loc[feat_lig, "atom_Index"][i],
                                "atom_number"] for i in range(len(ligand_profile.loc[
                                feat_lig, "atom_Index"]))])[1:-1])

                        if len(ligand_profile.loc[feat_lig, "atom_Index"]) > 1 :
                            atom_ligand_symbol.append(','.join([ligand.loc[
                                    ligand_profile.loc[feat_lig, "atom_Index"][i],
                                    "element_symbol"] for i in range(len(ligand_profile.loc[
                                    feat_lig, "atom_Index"]))]))
                        else :
                            atom_ligand_symbol.append(str([ligand.loc[
                                    ligand_profile.loc[feat_lig, "atom_Index"][i],
                                    "element_symbol"] for i in range(len(ligand_profile.loc[
                                    feat_lig, "atom_Index"]))])[2:-2])

                        if len(pocket_profile.loc[feat_pock, "atom_Index"]) > 1 :
                            aa.append(','.join(set([pocket.loc[
                                            pocket_profile.loc[
                                                feat_pock, "atom_Index"][i],
                                            'residue_name'] for i in range(
                                            len(pocket_profile.loc[
                                                    feat_pock, "atom_Index"]))])))

                            protein_atom_symbol.append(','.join([pocket.loc[
                                    pocket_profile.loc[feat_pock, "atom_Index"][i],
                                    'element_symbol'] for i in range(len(pocket_profile.loc[
                                        feat_pock, "atom_Index"]))]))

                            residues.append(','.join(set([str(pocket.loc[
                                pocket_profile.loc[feat_pock, "atom_Index"][i],
                                'residue_number']) + str(pocket.loc[
                                pocket_profile.loc[feat_pock, "atom_Index"][i],
                                'chain_id']) for i in range(len(pocket_profile.loc[
                                feat_pock, "atom_Index"]))])))
                        else :
                            aa.append(str(set([pocket.loc[
                                            pocket_profile.loc[
                                                feat_pock, "atom_Index"][i],
                                            'residue_name'] for i in range(
                                            len(pocket_profile.loc[
                                                    feat_pock, "atom_Index"]))]))[2:-2])

                            protein_atom_symbol.append(str(set([pocket.loc[
                                    pocket_profile.loc[feat_pock, "atom_Index"][i],
                                    'element_symbol'] for i in range(len(pocket_profile.loc[
                                        feat_pock, "atom_Index"]))]))[2:-2])

                            residues.append(str(set([str(pocket.loc[
                                pocket_profile.loc[feat_pock, "atom_Index"][i],
                                'residue_number']) + str(pocket.loc[
                                pocket_profile.loc[feat_pock, "atom_Index"][i],
                                'chain_id']) for i in range(len(pocket_profile.loc[
                                feat_pock, "atom_Index"]))]))[2:-2])


    results_hydrophobic_interactions = pd.DataFrame(list(zip(residues,
                                                             aa,
                                                             distances,
                                                             atom_ligand_id,
                                                             atom_ligand_symbol,
                                                             protein_atom_id,
                                                             protein_atom_symbol,
                                                             feat_lig_index,
                                                             feat_pock_index)),

                                        columns=["Residue", "AA", "Distance",
                                                 "N° Ligand Atom", "Ligand Atom Symbol",
                                                 "N° Protein Atom", "Protein Atom Symbol",
                                                 "Feat Lig Index", "Feat Pock Index"])

    results_hydrophobic_interactions.sort_values(by=['Residue'], inplace = True)
    return results_hydrophobic_interactions

def get_salt_bridges(ligand,
                     ligand_profile,
                     pocket,
                     pocket_profile,
                     salt_bridge_distance):
    """
    Computation of salt bridges between the ligand and the pocket.
    Parameters :
        Input :
            - ligand : dataframe of the ligand in pandas format from the Biopandas module.
            - ligand_profile : ligand in Mol File
            - pocket : dataframe of the pocket in pandas format from the Biopandas module.
            - pocket_profile : pocket in Mol File
            - salt_bridge_distance : float which defines the maximum distance between two
            opposited charged atoms group between the ligand and the pocket, for which we consider the
            salt bridges as valid.
        Output :
            - results_salt_bridges : salt bridges dataframe including the residues,
            residue numbers, distance, ligand and protein atom numbers, ligand and protein atom symbols,
            which are involved in each interaction.
    """   
    residues = []
    aa = []
    distances = []
    atom_ligand_id = []
    atom_ligand_symbol = []
    protein_atom_id = []
    protein_atom_symbol = []
    feat_lig_index = []
    feat_pock_index = []

    for feat_lig in ligand_profile.index :
        if ligand_profile.loc[feat_lig, 'Features_family'] == 'PosIonizable' :
            for feat_pock in pocket_profile.index :
                if pocket_profile.loc[feat_pock, 'Features_family'] == 'NegIonizable' :
                    distance = get_distance_between_2_points((ligand_profile.loc[feat_lig, "x_coord"],
                                                              ligand_profile.loc[feat_lig, "y_coord"],
                                                              ligand_profile.loc[feat_lig, "z_coord"]),
                                                             (pocket_profile.loc[feat_pock, "x_coord"], 
                                                              pocket_profile.loc[feat_pock, "y_coord"],
                                                              pocket_profile.loc[feat_pock, "z_coord"]))
                    if distance < salt_bridge_distance :

                        distances.append(round(float(distance), 2))
                        feat_lig_index.append(feat_lig)
                        feat_pock_index.append(feat_pock)

                        protein_atom_id.append(str(set([pocket.loc[
                                pocket_profile.loc[feat_pock, "atom_Index"][i],
                                'atom_number'] for i in range(len(pocket_profile.loc[
                                    feat_pock, "atom_Index"]))]))[1:-1])

                        atom_ligand_id.append(str([ligand.loc[
                                ligand_profile.loc[feat_lig, "atom_Index"][i],
                                "atom_number"] for i in range(len(ligand_profile.loc[
                                feat_lig, "atom_Index"]))])[1:-1])

                        if len(ligand_profile.loc[feat_lig, "atom_Index"]) > 1 :
                            atom_ligand_symbol.append(','.join([ligand.loc[
                                    ligand_profile.loc[feat_lig, "atom_Index"][i],
                                    "element_symbol"] for i in range(len(ligand_profile.loc[
                                    feat_lig, "atom_Index"]))]))
                        else :
                            atom_ligand_symbol.append(str([ligand.loc[
                                    ligand_profile.loc[feat_lig, "atom_Index"][i],
                                    "element_symbol"] for i in range(len(ligand_profile.loc[
                                    feat_lig, "atom_Index"]))])[2:-2])

                        if len(pocket_profile.loc[feat_pock, "atom_Index"]) > 1 :
                            aa.append(','.join(set([pocket.loc[
                                            pocket_profile.loc[
                                                feat_pock, "atom_Index"][i],
                                            'residue_name'] for i in range(
                                            len(pocket_profile.loc[
                                                    feat_pock, "atom_Index"]))])))

                            protein_atom_symbol.append(','.join([pocket.loc[
                                    pocket_profile.loc[feat_pock, "atom_Index"][i],
                                    'element_symbol'] for i in range(len(pocket_profile.loc[
                                        feat_pock, "atom_Index"]))]))

                            residues.append(','.join(set([str(pocket.loc[
                                pocket_profile.loc[feat_pock, "atom_Index"][i],
                                'residue_number']) + str(pocket.loc[
                                pocket_profile.loc[feat_pock, "atom_Index"][i],
                                'chain_id']) for i in range(len(pocket_profile.loc[
                                feat_pock, "atom_Index"]))])))
                        else :
                            aa.append(str(set([pocket.loc[
                                            pocket_profile.loc[
                                                feat_pock, "atom_Index"][i],
                                            'residue_name'] for i in range(
                                            len(pocket_profile.loc[
                                                    feat_pock, "atom_Index"]))]))[2:-2])

                            protein_atom_symbol.append(str(set([pocket.loc[
                                    pocket_profile.loc[feat_pock, "atom_Index"][i],
                                    'element_symbol'] for i in range(len(pocket_profile.loc[
                                        feat_pock, "atom_Index"]))]))[2:-2])

                            residues.append(str(set([str(pocket.loc[
                                pocket_profile.loc[feat_pock, "atom_Index"][i],
                                'residue_number']) + str(pocket.loc[
                                pocket_profile.loc[feat_pock, "atom_Index"][i],
                                'chain_id']) for i in range(len(pocket_profile.loc[
                                feat_pock, "atom_Index"]))]))[2:-2])

    for feat_lig in ligand_profile.index :
        if ligand_profile.loc[feat_lig, 'Features_family'] == 'NegIonizable' :
            for feat_pock in pocket_profile.index :
                if pocket_profile.loc[feat_pock, 'Features_family'] == 'PosIonizable' :
                    if not (pocket_profile.loc[feat_pock, 'Features_type'] == 'PosN'
                            and pocket_profile.loc[feat_pock, 'atom_Index'][0][0:3] == 'ARG') :
                        distance = get_distance_between_2_points((ligand_profile.loc[feat_lig, "x_coord"],
                                                                  ligand_profile.loc[feat_lig, "y_coord"],
                                                                  ligand_profile.loc[feat_lig, "z_coord"]),
                                                                 (pocket_profile.loc[feat_pock, "x_coord"], 
                                                                  pocket_profile.loc[feat_pock, "y_coord"],
                                                                  pocket_profile.loc[feat_pock, "z_coord"]))
                        if distance < salt_bridge_distance :

                            distances.append(round(float(distance), 2))
                            feat_lig_index.append(feat_lig)
                            feat_pock_index.append(feat_pock)

                            protein_atom_id.append(str(set([pocket.loc[
                                    pocket_profile.loc[feat_pock, "atom_Index"][i],
                                    'atom_number'] for i in range(len(pocket_profile.loc[
                                        feat_pock, "atom_Index"]))]))[1:-1])

                            atom_ligand_id.append(str([ligand.loc[
                                    ligand_profile.loc[feat_lig, "atom_Index"][i],
                                    "atom_number"] for i in range(len(ligand_profile.loc[
                                    feat_lig, "atom_Index"]))])[1:-1])

                            if len(ligand_profile.loc[feat_lig, "atom_Index"]) > 1 :
                                atom_ligand_symbol.append(','.join([ligand.loc[
                                        ligand_profile.loc[feat_lig, "atom_Index"][i],
                                        "element_symbol"] for i in range(len(ligand_profile.loc[
                                        feat_lig, "atom_Index"]))]))
                            else :
                                atom_ligand_symbol.append(str([ligand.loc[
                                        ligand_profile.loc[feat_lig, "atom_Index"][i],
                                        "element_symbol"] for i in range(len(ligand_profile.loc[
                                        feat_lig, "atom_Index"]))])[2:-2])

                            if len(pocket_profile.loc[feat_pock, "atom_Index"]) > 1 :
                                aa.append(','.join(set([pocket.loc[
                                                pocket_profile.loc[
                                                    feat_pock, "atom_Index"][i],
                                                'residue_name'] for i in range(
                                                len(pocket_profile.loc[
                                                        feat_pock, "atom_Index"]))])))

                                protein_atom_symbol.append(','.join([pocket.loc[
                                        pocket_profile.loc[feat_pock, "atom_Index"][i],
                                        'element_symbol'] for i in range(len(pocket_profile.loc[
                                            feat_pock, "atom_Index"]))]))

                                residues.append(','.join(set([str(pocket.loc[
                                    pocket_profile.loc[feat_pock, "atom_Index"][i],
                                    'residue_number']) + str(pocket.loc[
                                    pocket_profile.loc[feat_pock, "atom_Index"][i],
                                    'chain_id']) for i in range(len(pocket_profile.loc[
                                    feat_pock, "atom_Index"]))])))
                            else :
                                aa.append(str(set([pocket.loc[
                                                pocket_profile.loc[
                                                    feat_pock, "atom_Index"][i],
                                                'residue_name'] for i in range(
                                                len(pocket_profile.loc[
                                                        feat_pock, "atom_Index"]))]))[2:-2])

                                protein_atom_symbol.append(str(set([pocket.loc[
                                        pocket_profile.loc[feat_pock, "atom_Index"][i],
                                        'element_symbol'] for i in range(len(pocket_profile.loc[
                                            feat_pock, "atom_Index"]))]))[2:-2])

                                residues.append(str(set([str(pocket.loc[
                                    pocket_profile.loc[feat_pock, "atom_Index"][i],
                                    'residue_number']) + str(pocket.loc[
                                    pocket_profile.loc[feat_pock, "atom_Index"][i],
                                    'chain_id']) for i in range(len(pocket_profile.loc[
                                    feat_pock, "atom_Index"]))]))[2:-2])

    results_salt_bridges = pd.DataFrame(list(zip(residues,
                                                 aa,
                                                 distances,
                                                 atom_ligand_id,
                                                 atom_ligand_symbol,
                                                 protein_atom_id,
                                                 protein_atom_symbol,
                                                 feat_lig_index,
                                                 feat_pock_index)),

                                        columns=["Residue", "AA", "Distance",
                                                 "N° Ligand Atom", "Ligand Atom Symbol",
                                                 "N° Protein Atom", "Protein Atom Symbol",
                                                 "Feat Lig Index", "Feat Pock Index"])

    results_salt_bridges.sort_values(by=['Residue'], inplace = True)
    return results_salt_bridges

def get_h_bonds_interactions(ligand,
                             ligand_profile,
                             pocket,
                             pocket_profile,
                             h_bond_distance,
                             h_bond_angle_sp3,
                             h_bond_angle_sp2,
                             salt_bridges_results):
    """
    Computation of hydrogen bonds between the ligand and the pocket.
    Parameters :
        Input :
            - ligand : dataframe of the ligand in pandas format from the Biopandas module.
            - ligand_profile : ligand in Mol File
            - pocket : dataframe of the pocket in pandas format from the Biopandas module.
            - pocket_profile : pocket in Mol File
            - h_bond_distance : float which defines the maximum distance between hydrogen bond donor
            and one hydrogen bond acceptor atoms between the ligand and the pocket, for which we consider the
            hydrogen bond interaction as valid.
            - h_bond_angle_sp3 : Maximum deviation of the angle consisting of the carbon bearing
            the donor heteroatom, the donor heteroatom and the acceptor heteroatom.
            - h_bond_angle_sp2 : Maximum deviation of the angle consisting of the donor heteroatom,
            the hydrogen and the acceptor heteroatom.
            - salt_bridges_results : salt bridges dataframe results. If 2 atoms between the ligand and
            the protein have a valid hydrogen bond AND a valid salt bridge, the hydrogen bond is cancel
            in favor of the salt bridge.
        Output :
            - results_h_bond_donor : hydrogen bond donor interactions dataframe including the residues,
            residue numbers, distance, ligand and protein atom numbers, ligand and protein atom symbols,
            which are involved in each interaction.
            - results_h_bond_acceptor : hydrogen bond acceptor interactions dataframe including the residues,
            residue numbers, distance, ligand and protein atom numbers, ligand and protein atom symbols,
            which are involved in each interaction. 
    """   
    residues = []
    aa = []
    distances = []
    atom_ligand_id = []
    atom_ligand_symbol = []
    protein_atom_id = []
    protein_atom_symbol = []
    feat_lig_index = []
    feat_pock_index = []


    for feat_lig in ligand_profile.index :
        if ligand_profile.loc[feat_lig, 'Features_family'] == 'Donor' :
            atom_L = ligand_profile.loc[feat_lig,'atom_Index'][0]
            
            if 'H' in [ligand.loc[
                neighbor,'Atom Symbol'] for neighbor in ligand.loc[
                    atom_L,'Neighbors']]:
                
                for feat_pock in pocket_profile.index :
                    if pocket_profile.loc[feat_pock, 'Features_family'] == 'Acceptor' :
                        atom_P = pocket_profile.loc[feat_pock,'atom_Index'][0]
                        
                        _pass = 0
                        for index in salt_bridges_results.index :
                            if (str(ligand.loc[atom_L, 'atom_number']) in salt_bridges_results.loc[
                                    index, 'N° Ligand Atom'].split(', ')  and
                                str(pocket.loc[atom_P, 'atom_number']) in salt_bridges_results.loc[
                                    index, 'N° Protein Atom'].split(', ')) :
                                _pass += 1
                                
                        if _pass == 0 :
                            distance = h_bond_distance + 1
                            if (ligand.loc[atom_L, 'Hybridation'] == 'SP3'
                                or ('O' in ligand.loc[atom_L, 'element_symbol']
                                    and 'H' in [ligand.loc[neighbor, 'Atom Symbol'] 
                                                for neighbor in ligand.loc[atom_L, 'Neighbors']])):

                                if ligand.loc[atom_L, 'Hybridation'] == 'SP3' :
                                    angle_target = 109.5
                                elif ('O' in ligand.loc[atom_L, 'element_symbol']
                                    and 'H' in [ligand.loc[neighbor, 'Atom Symbol'] 
                                                for neighbor in ligand.loc[atom_L, 'Neighbors']]):
                                    angle_target = 120

                                C_neighbor = (
                                    [ligand.loc[neighbor].name for neighbor in
                                     ligand.loc[atom_L,'Neighbors'] 
                                     if ligand.loc[neighbor,'Atom Symbol'] == 'C'][0]
                                )

                                angle = get_angle(ligand.loc[atom_L],
                                              ligand.loc[C_neighbor],
                                              pocket.loc[atom_P])
                                if (angle > (angle_target - h_bond_angle_sp3) and
                                    angle < (angle_target + h_bond_angle_sp3)):


                                    distance = get_distance_between_2_points(
                                        (ligand_profile.loc[feat_lig, "x_coord"],
                                         ligand_profile.loc[feat_lig, "y_coord"],
                                         ligand_profile.loc[feat_lig, "z_coord"]),
                                        (pocket_profile.loc[feat_pock, "x_coord"], 
                                         pocket_profile.loc[feat_pock, "y_coord"],
                                         pocket_profile.loc[feat_pock, "z_coord"]))

                            elif ligand.loc[atom_L, 'Hybridation'] == 'SP2' :
                                H_neighbor = ([ligand.loc[neighbor].name for neighbor in
                                     ligand.loc[atom_L,'Neighbors'] 
                                     if ligand.loc[neighbor,'Atom Symbol'] == 'H'][0])

                                angle = get_angle(ligand.loc[H_neighbor],
                                              ligand.loc[atom_L],
                                              pocket.loc[atom_P])
                                if angle > (180 - h_bond_angle_sp2) :
                                    distance = get_distance_between_2_points(
                                        (ligand_profile.loc[feat_lig, "x_coord"],
                                         ligand_profile.loc[feat_lig, "y_coord"],
                                         ligand_profile.loc[feat_lig, "z_coord"]),
                                        (pocket_profile.loc[feat_pock, "x_coord"], 
                                         pocket_profile.loc[feat_pock, "y_coord"],
                                         pocket_profile.loc[feat_pock, "z_coord"]))

                            if distance < h_bond_distance :

                                distances.append(round(float(distance), 2))
                                feat_lig_index.append(feat_lig)
                                feat_pock_index.append(feat_pock)

                                protein_atom_id.append(str(set([pocket.loc[
                                        pocket_profile.loc[feat_pock, "atom_Index"][i],
                                        'atom_number'] for i in range(len(pocket_profile.loc[
                                            feat_pock, "atom_Index"]))]))[1:-1])

                                atom_ligand_id.append(str([ligand.loc[
                                        ligand_profile.loc[feat_lig, "atom_Index"][i],
                                        "atom_number"] for i in range(len(ligand_profile.loc[
                                        feat_lig, "atom_Index"]))])[1:-1])

                                if len(ligand_profile.loc[feat_lig, "atom_Index"]) > 1 :
                                    atom_ligand_symbol.append(','.join([ligand.loc[
                                            ligand_profile.loc[feat_lig, "atom_Index"][i],
                                            "element_symbol"] for i in range(len(ligand_profile.loc[
                                            feat_lig, "atom_Index"]))]))
                                else :
                                    atom_ligand_symbol.append(str([ligand.loc[
                                            ligand_profile.loc[feat_lig, "atom_Index"][i],
                                            "element_symbol"] for i in range(len(ligand_profile.loc[
                                            feat_lig, "atom_Index"]))])[2:-2])

                                if len(pocket_profile.loc[feat_pock, "atom_Index"]) > 1 :
                                    aa.append(','.join(set([pocket.loc[
                                                    pocket_profile.loc[
                                                        feat_pock, "atom_Index"][i],
                                                    'residue_name'] for i in range(
                                                    len(pocket_profile.loc[
                                                            feat_pock, "atom_Index"]))])))

                                    protein_atom_symbol.append(','.join([pocket.loc[
                                            pocket_profile.loc[feat_pock, "atom_Index"][i],
                                            'element_symbol'] for i in range(len(pocket_profile.loc[
                                                feat_pock, "atom_Index"]))]))

                                    residues.append(','.join(set([str(pocket.loc[
                                        pocket_profile.loc[feat_pock, "atom_Index"][i],
                                        'residue_number']) + str(pocket.loc[
                                        pocket_profile.loc[feat_pock, "atom_Index"][i],
                                        'chain_id']) for i in range(len(pocket_profile.loc[
                                        feat_pock, "atom_Index"]))])))
                                else :
                                    aa.append(str(set([pocket.loc[
                                                    pocket_profile.loc[
                                                        feat_pock, "atom_Index"][i],
                                                    'residue_name'] for i in range(
                                                    len(pocket_profile.loc[
                                                            feat_pock, "atom_Index"]))]))[2:-2])

                                    protein_atom_symbol.append(str(set([pocket.loc[
                                            pocket_profile.loc[feat_pock, "atom_Index"][i],
                                            'element_symbol'] for i in range(len(pocket_profile.loc[
                                                feat_pock, "atom_Index"]))]))[2:-2])

                                    residues.append(str(set([str(pocket.loc[
                                        pocket_profile.loc[feat_pock, "atom_Index"][i],
                                        'residue_number']) + str(pocket.loc[
                                        pocket_profile.loc[feat_pock, "atom_Index"][i],
                                        'chain_id']) for i in range(len(pocket_profile.loc[
                                        feat_pock, "atom_Index"]))]))[2:-2])                

    results_h_bond_donor = pd.DataFrame(list(zip(residues,
                                                 aa,
                                                 distances,
                                                 atom_ligand_id,
                                                 atom_ligand_symbol,
                                                 protein_atom_id,
                                                 protein_atom_symbol,
                                                 feat_lig_index,
                                                 feat_pock_index)),

                                        columns=["Residue", "AA", "Distance",
                                                 "N° Ligand Atom", "Ligand Atom Symbol",
                                                 "N° Protein Atom", "Protein Atom Symbol",
                                                 "Feat Lig Index", "Feat Pock Index"])

    results_h_bond_donor.sort_values(by=['Residue'], inplace = True)

    residues = []
    aa = []
    distances = []
    atom_ligand_id = []
    atom_ligand_symbol = []
    protein_atom_id = []
    protein_atom_symbol = []
    feat_lig_index = []
    feat_pock_index = []

    for feat_lig in ligand_profile.index :
        if ligand_profile.loc[feat_lig, 'Features_family'] == 'Acceptor' :
            atom_L = ligand_profile.loc[feat_lig,'atom_Index'][0]
            
            for feat_pock in pocket_profile.index :
                if pocket_profile.loc[feat_pock, 'Features_family'] == 'Donor' :
                    atom_P = pocket_profile.loc[feat_pock,'atom_Index'][0]
                    
                    _pass = 0
                    for index in salt_bridges_results.index :
                        if (str(ligand.loc[atom_L, 'atom_number']) in salt_bridges_results.loc[
                                index, 'N° Ligand Atom'].split(', ')  and
                            str(pocket.loc[atom_P, 'atom_number']) in salt_bridges_results.loc[
                                index, 'N° Protein Atom'].split(', ')) :
                            _pass += 1
                    
                    if _pass == 0 :
                        distance = h_bond_distance + 1
                        if 'H' in [pocket.loc[
                            neighbor,'Atom Symbol'] for neighbor in pocket.loc[
                            atom_P,'Neighbors']]:
                            if (pocket.loc[atom_P, 'Hybridation'] == 'SP3'
                                or ('O' in pocket.loc[atom_P, 'element_symbol']
                                    and 'H' in [pocket.loc[neighbor, 'Atom Symbol'] 
                                                for neighbor in pocket.loc[atom_P, 'Neighbors']])):

                                if pocket.loc[atom_P, 'Hybridation'] == 'SP3' :
                                    angle_target = 109.5
                                elif ('O' in pocket.loc[atom_P, 'element_symbol']
                                    and 'H' in [pocket.loc[neighbor, 'Atom Symbol'] 
                                                for neighbor in pocket.loc[atom_P, 'Neighbors']]):
                                    angle_target = 120

                                C_neighbor = (
                                    [pocket.loc[neighbor].name for neighbor in
                                     pocket.loc[atom_P,'Neighbors'] 
                                     if pocket.loc[neighbor,'Atom Symbol'] == 'C'][0]
                                )

                                angle = get_angle(pocket.loc[atom_P],
                                              pocket.loc[C_neighbor],
                                              ligand.loc[atom_L])
                                if (angle > (angle_target - h_bond_angle_sp3) and 
                                    angle < (angle_target + h_bond_angle_sp3)):

                                    distance = get_distance_between_2_points(
                                        (ligand_profile.loc[feat_lig, "x_coord"],
                                         ligand_profile.loc[feat_lig, "y_coord"],
                                         ligand_profile.loc[feat_lig, "z_coord"]),
                                        (pocket_profile.loc[feat_pock, "x_coord"], 
                                         pocket_profile.loc[feat_pock, "y_coord"],
                                         pocket_profile.loc[feat_pock, "z_coord"]))

                            elif pocket.loc[atom_P, 'Hybridation'] == 'SP2' :
                                H_neighbor = (
                                    [pocket.loc[neighbor].name for neighbor in
                                     pocket.loc[atom_P,'Neighbors'] 
                                     if pocket.loc[neighbor,'Atom Symbol'] == 'H'][0]
                                )

                                angle = get_angle(pocket.loc[H_neighbor],
                                              pocket.loc[atom_P],
                                              ligand.loc[atom_L])
                                if angle > (180 - h_bond_angle_sp2) :
                                    distance = get_distance_between_2_points(
                                        (ligand_profile.loc[feat_lig, "x_coord"],
                                         ligand_profile.loc[feat_lig, "y_coord"],
                                         ligand_profile.loc[feat_lig, "z_coord"]),
                                        (pocket_profile.loc[feat_pock, "x_coord"], 
                                         pocket_profile.loc[feat_pock, "y_coord"],
                                         pocket_profile.loc[feat_pock, "z_coord"]))

                            if distance < h_bond_distance :

                                distances.append(round(float(distance), 2))
                                feat_lig_index.append(feat_lig)
                                feat_pock_index.append(feat_pock)

                                protein_atom_id.append(str(set([pocket.loc[
                                        pocket_profile.loc[feat_pock, "atom_Index"][i],
                                        'atom_number'] for i in range(len(pocket_profile.loc[
                                            feat_pock, "atom_Index"]))]))[1:-1])

                                atom_ligand_id.append(str([ligand.loc[
                                        ligand_profile.loc[feat_lig, "atom_Index"][i],
                                        "atom_number"] for i in range(len(ligand_profile.loc[
                                        feat_lig, "atom_Index"]))])[1:-1])

                                if len(ligand_profile.loc[feat_lig, "atom_Index"]) > 1 :
                                    atom_ligand_symbol.append(','.join([ligand.loc[
                                            ligand_profile.loc[feat_lig, "atom_Index"][i],
                                            "element_symbol"] for i in range(len(ligand_profile.loc[
                                            feat_lig, "atom_Index"]))]))
                                else :
                                    atom_ligand_symbol.append(str([ligand.loc[
                                            ligand_profile.loc[feat_lig, "atom_Index"][i],
                                            "element_symbol"] for i in range(len(ligand_profile.loc[
                                            feat_lig, "atom_Index"]))])[2:-2])

                                if len(pocket_profile.loc[feat_pock, "atom_Index"]) > 1 :
                                    aa.append(','.join(set([pocket.loc[
                                                    pocket_profile.loc[
                                                        feat_pock, "atom_Index"][i],
                                                    'residue_name'] for i in range(
                                                    len(pocket_profile.loc[
                                                            feat_pock, "atom_Index"]))])))

                                    protein_atom_symbol.append(','.join([pocket.loc[
                                            pocket_profile.loc[feat_pock, "atom_Index"][i],
                                            'element_symbol'] for i in range(len(pocket_profile.loc[
                                                feat_pock, "atom_Index"]))]))

                                    residues.append(','.join(set([str(pocket.loc[
                                        pocket_profile.loc[feat_pock, "atom_Index"][i],
                                        'residue_number']) + str(pocket.loc[
                                        pocket_profile.loc[feat_pock, "atom_Index"][i],
                                        'chain_id']) for i in range(len(pocket_profile.loc[
                                        feat_pock, "atom_Index"]))])))
                                else :
                                    aa.append(str(set([pocket.loc[
                                                    pocket_profile.loc[
                                                        feat_pock, "atom_Index"][i],
                                                    'residue_name'] for i in range(
                                                    len(pocket_profile.loc[
                                                            feat_pock, "atom_Index"]))]))[2:-2])

                                    protein_atom_symbol.append(str(set([pocket.loc[
                                            pocket_profile.loc[feat_pock, "atom_Index"][i],
                                            'element_symbol'] for i in range(len(pocket_profile.loc[
                                                feat_pock, "atom_Index"]))]))[2:-2])

                                    residues.append(str(set([str(pocket.loc[
                                        pocket_profile.loc[feat_pock, "atom_Index"][i],
                                        'residue_number']) + str(pocket.loc[
                                        pocket_profile.loc[feat_pock, "atom_Index"][i],
                                        'chain_id']) for i in range(len(pocket_profile.loc[
                                        feat_pock, "atom_Index"]))]))[2:-2])                

    results_h_bond_acceptor = pd.DataFrame(list(zip(residues,
                                                 aa,
                                                 distances,
                                                 atom_ligand_id,
                                                 atom_ligand_symbol,
                                                 protein_atom_id,
                                                 protein_atom_symbol,
                                                 feat_lig_index,
                                                 feat_pock_index)),

                                        columns=["Residue", "AA", "Distance",
                                                 "N° Ligand Atom", "Ligand Atom Symbol",
                                                 "N° Protein Atom", "Protein Atom Symbol",
                                                 "Feat Lig Index", "Feat Pock Index"])

    results_h_bond_acceptor.sort_values(by=['Residue'], inplace = True)
    return results_h_bond_donor, results_h_bond_acceptor       



class Profiler :
    
    def __init__(self, 
                 size_pocket = 5.5,
                 hydrophobic_bond_distance = 4.0,
                 h_bond_distance = 4.1,
                 h_bond_angle_sp3 = 19.5,
                 h_bond_angle_sp2 = 34,
                 salt_bridge_distance = 5.5) :
        """
        Profiler is a class that allows the 3D pharmacophore of a protein-ligand complex to be established. 
        From a PDB code or a pathway, it extracts the ligand as well as the residues that constitute the 
        pocket. Profiler then builds the pharmacophore profile of the ligand and the pocket. 
        Finally, Profiler compares the two profiles to determine the electrostatic interactions that
        that would allow the complex to be stabilised.

        
        The initialization of this class includes the definition of several parameters :
                 - size_pocket : float which defines the maximum distance between a ligand atom and a protein atom
            of the protein, for which the corresponding residue is considered as constituting the pocket.,
                 - hydrophobic_bond_distance : float which defines the maximum distance between two
            hydrophobic atoms between the ligand and the pocket, for which we consider the
            hydrophobic interaction as valid.,
                 - h_bond_distance = float which defines the maximum distance between hydrogen bond donor
            and one hydrogen bond acceptor atoms between the ligand and the pocket, for which we consider the
            hydrogen bond interaction as valid.
                 - h_bond_angle_sp3 : Maximum deviation of the angle consisting of the carbon bearing
            the donor heteroatom, the donor heteroatom and the acceptor heteroatom.
                 - h_bond_angle_sp2 : Maximum deviation of the angle consisting of the donor heteroatom,
            the hydrogen and the acceptor heteroatom.
                 - salt_bridge_distance = float which defines the maximum distance between two
            opposited charged atoms group between the ligand and the pocket, for which we consider the
            salt bridges as valid. :
        """
        
        self.size_pocket = size_pocket
        self.hydrophobic_bond_distance = hydrophobic_bond_distance
        self.h_bond_distance = h_bond_distance
        self.h_bond_angle_sp3 = h_bond_angle_sp3
        self.h_bond_angle_sp2 = h_bond_angle_sp2
        self.salt_bridge_distance = salt_bridge_distance

    def get_complex(self, protein=None, ligand=None, chain=None) :
        """
        Preparation phase to obtain the ligand and the pocket and to compute 
        the phamacophoric profile of each respectively.
        Parameters :
            Input :
                - protein : PDB code or pathway (string)
                - ligand : PDB code or pathway (string)
                - chain : Letter(s) corresponding to the channel(s) to be selected (string or list)
        """   
        if len(protein) >= 4 :
            if len(protein) == 4 :
                complexe = PandasPdb().fetch_pdb(protein) #Charge la protein avec le module Biopandas

            elif len(protein) > 4 :
                complexe = PandasPdb().read_pdb(protein)

            complex_without_water = complexe
            complex_without_water.df['HETATM'].drop(
                index=complex_without_water.df['HETATM'][
                    complex_without_water.df['HETATM']['residue_name'] == 'HOH'].index,
                inplace=True)
            
            if not ligand and complex_without_water.df['HETATM'].empty :
                print("Error ! There isn't any ligand in the input protein.",
                      " Please check your input protein or input an external ligand.")
            else :
                if chain or len(set(complexe.df['ATOM']['chain_id'])) == 1 :
                    if len(set(complexe.df['ATOM']['chain_id'])) == 1 :
                        chain = set(complexe.df['ATOM']['chain_id'])
                    
                    g = complexe.df['ATOM'].groupby('chain_id')
                    complexe.df['ATOM'] = pd.concat([g.get_group(i.upper()) for i in chain],
                                                     axis=0,
                                                     verify_integrity=True)
                    g = complexe.df['HETATM'].groupby('chain_id')
                    complexe.df['HETATM'] = pd.concat([g.get_group(i.upper()) for i in chain],
                                                     axis=0,
                                                     verify_integrity=True)
               
                    complexe.to_pdb(path='complex.pdb')
                    p1 = pymol2.PyMOL()
                    p1.start()
                    p1.cmd.load('complex.pdb', 'mol')
                    p1.cmd.h_add()
                    p1.cmd.save('complex_H.pdb')
                    p1.stop()

                    self.complexe = PandasPdb().read_pdb('complex_H.pdb').df

                    self.protein = self.complexe['ATOM'] #Définis la protéine
                    protein_initialization(self.protein) #Initialise la protéine

                    if not ligand and len(set(complex_without_water.df['HETATM']['residue_name'])) > 1 :
                        print('Error ! You have to choose one ligand. Either you input an external ligand,',
                              'either you select one ligand among the ones in the input protein :\n')
                        print(set(complex_without_water.df['HETATM']['residue_name']))
                        
                    else:
                        if not ligand and len(set(complex_without_water.df['HETATM']['residue_name'])) == 1 :
                            ligand = str(set(complex_without_water.df['HETATM']['residue_name']))[2:-2]
                        if len(ligand) == 3 :
                            if len(set(self.complexe['HETATM'][
                                self.complexe[
                                    'HETATM']['residue_name'] == ligand.upper()]['residue_name'][1:] +
                                       pd.Series(str(self.complexe['HETATM'][
                                           self.complexe['HETATM'][
                                               'residue_name'] == ligand.upper()].residue_number.to_list())[
                                           1:-1].split(','), index = self.complexe[
                                           'HETATM'][self.complexe['HETATM'][
                                           'residue_name'] == ligand.upper()].index)[1:])) > 1 :
                                
                                print(f"The input protein include several ligand named '{ligand.upper()}' among",
                                      'the selected chains. Only the first one on the list was considered.')

                            lig = complexe
                            if len(chain) == 1 :
                                lig.df['HETATM'] = lig.df[
                                    'HETATM'].groupby('chain_id').get_group(list(chain.upper())[0])
                            else :
                                lig.df['HETATM'] = lig.df['HETATM'].groupby('chain_id').get_group(chain[0].upper())
                            lig.df['HETATM'] = lig.df[
                                'HETATM'][lig.df['HETATM']['residue_name'] == ligand.upper()]
                            del lig.df['ATOM']
                            del lig.df['ANISOU']
                            lig.to_pdb(path='ligand_biopandas.pdb')
                            with open('ligand_biopandas.pdb', "r") as f:
                                content = f.read()
                                f.close()

                            with open('ligand_biopandas.pdb', 'w') as output_file:
                                for line in content.strip().split("\n") :
                                    if line[0:6] == 'HETATM':
                                        if line[-2] != ' ':
                                            output_file.write(line[:-7] + '' + line[-6:] + ' ' + "\n")
                                        else :
                                            output_file.write(line + "\n")
                                    else :
                                        output_file.write(line + "\n")
                            
                            if len(protein) == 4 :
                                pdb_ligands = get_pdb_ligands(protein)
                                smiles_ligand = pdb_ligands[ligand.upper()]['SMILES (CACTVS)']
                                self.ligand_structure = get_ligand_with_smiles(
                                    'ligand_biopandas.pdb', smiles_ligand)
                            elif len(protein) > 4 :
                                self.ligand_structure = get_ligand_with_pymol('ligand_biopandas.pdb', ligand)
                                os.remove('pymol_ligand.mol')
                            
                            os.remove('ligand_biopandas.pdb')
                            Chem.MolToPDBFile(self.ligand_structure, 'RDKit_ligand.pdb')
                            self.ligand = PandasPdb().read_pdb('RDKit_ligand.pdb').df['HETATM']
                            ligand_initialization(self.ligand)
                            self.ligand, self.ligand_profile = caracterisation(self.ligand,
                                                                               self.ligand_structure)

                        elif len(ligand) > 3 :
                            if ligand[-3:] == 'pdb' :
                                self.ligand_structure = Chem.AddHs(Chem.MolFromPDBFile(ligand), addCoords=True)
                            elif ligand[-3:] == 'mol':
                                self.ligand_structure = Chem.AddHs(Chem.MolFromMolFile(ligand), addCoords=True)
                            Chem.MolToPDBFile(self.ligand_structure, 'ligand.pdb')
                            self.ligand = PandasPdb().read_pdb('ligand.pdb').df['HETATM']
                            ligand_initialization(self.ligand)
                            self.ligand, self.ligand_profile = caracterisation(self.ligand,
                                                                               self.ligand_structure)

                        self.pocket, self.pocket_structure = get_pocket(self.protein,
                                                                    self.ligand,
                                                                    self.size_pocket)
                        
                        self.pocket, self.pocket_profile = caracterisation(self.pocket,
                                                                           self.pocket_structure,
                                                                           type_molecule = 'Pocket')
                
                else :
                    print('The input protein include several chains. Which ones do you want to keep ?\n')
                    for i in set(complexe.df['ATOM']['chain_id']) :
                        print(i)
                    print('\nFor informations the associated ligands are :\n')
                    for i in set(complexe.df['HETATM']['residue_name'][1:] +
            ' number' + pd.Series(str(complexe.df['HETATM'].residue_number.to_list())[1:-1].split(','),
              index = complexe.df['HETATM'].index)[1:] + ' chain ' + complexe.df['HETATM']['chain_id'][1:]) :
                        if 'HOH' not in i :
                            print(i)
                        
        else :
            print("Error ! The code or pathway of the input protein is wrong or does not exist.")
            

    def get_interactions(self, view=False) :
        """
        Determines electrostatic interactions (hydrophobic, hydrogen bonding and salt bridge) 
        from the pharmacophoric profiles of the ligand and the pocket.
        Parameters :
            Input :
                - view : if toggled into 'True', at the end of the interaction calculation,
                a PyMOL window is launched with a visualisation of the 3D pharmacophore.
        """
        
        self.hydrophobic_interactions = get_hydrophobic_interactions(self.ligand,
                                                                     self.ligand_profile,
                                                                     self.pocket,
                                                                     self.pocket_profile,
                                                                     self.hydrophobic_bond_distance)
        
        self.salt_bridges_interactions = get_salt_bridges(self.ligand,
                                                            self.ligand_profile,
                                                            self.pocket,
                                                            self.pocket_profile,
                                                            self.salt_bridge_distance)
        
        self.h_bonds_donor, self.h_bonds_acceptor = get_h_bonds_interactions(self.ligand,
                                                                             self.ligand_profile,
                                                                             self.pocket,
                                                                             self.pocket_profile,
                                                                             self.h_bond_distance,
                                                                             self.h_bond_angle_sp3,
                                                                             self.h_bond_angle_sp2,
                                                                             self.salt_bridges_interactions)
        
        if self.hydrophobic_interactions.empty :
            print("There is no hydrophobic interaction.")
        else :
            print('Hydrophobic Interactions'.upper(),'\n')
            print(self.hydrophobic_interactions.iloc[:, 0:7], '\n')
        if self.h_bonds_donor.empty :
            print("There is no donor hydrogen bond.\n")
        else :
            print('Donor Hydrogen Bonds'.upper(), '\n')
            print(self.h_bonds_donor.iloc[:, 0:7], '\n')
        if self.h_bonds_acceptor.empty :
            print("There is no acceptor hydrogen bond.\n")
        else :
            print('Acceptor Hydrogen Bonds'.upper(), '\n')
            print(self.h_bonds_acceptor.iloc[:, 0:7], '\n')
        if self.salt_bridges_interactions.empty :
            print("There is no salt bridges.\n")
        else :
            print('Salt Bridges'.upper(), '\n')
            print(self.salt_bridges_interactions.iloc[:, 0:7], '\n')
        
        with open('profiler.pml', 'w') as w :
            w.write('load ./pocket.pdb, pocket \n' +
                    'run cgo_arrow.py \n' +
                    'show_as lines, pocket \n' +
                    'label n. CA, "%s %s" % (resn, resi) \n' +
                    'color slate, pocket and (name C*) \n' +
                    'load ./RDKit_ligand.pdb, ligand \n' +
                    'show_as stick, ligand \n' +
                    'color white, ligand and (name C*) \n' +
                    'set sphere_color, yellow \n' +
                    'set sphere_transparency,0.5 \n' +
                    'set sphere_scale, 0.3 \n' +
                    'load ./complex.pdb, complex \n' +
                    'show_as cartoon, complex \n'+
                    'set cartoon_transparency, 0.75 \n' +
                    'color lightpink, complex and (name C*) \n')
            
            for i in range(len(self.hydrophobic_interactions)):
                w.write(f"show sphere, (ligand and id {self.hydrophobic_interactions.loc[i, 'N° Ligand Atom']}) \n")
                w.write(f"show sphere, (pocket and id {self.hydrophobic_interactions.loc[i, 'N° Protein Atom']}) \n")
            
            for i in range(len(self.h_bonds_donor)):
                w.write(f"cgo_arrow (ligand and id {self.h_bonds_donor.loc[i, 'N° Ligand Atom']})," + 
                f" (pocket and id {self.h_bonds_donor.loc[i, 'N° Protein Atom']})," + 
                " gap=0.1, hradius=0.25, hlength=0.5,  radius=0.075, color=cyan \n")
            
            for i in range(len(self.h_bonds_acceptor)):
                w.write(f"cgo_arrow (pocket and id {self.h_bonds_acceptor.loc[i, 'N° Protein Atom']})," + 
                f" (ligand and id {self.h_bonds_acceptor.loc[i, 'N° Ligand Atom']})," + 
                " gap=0.1, hradius=0.25, hlength=0.5,  radius=0.075, color=purple \n")
            
            for i in range(len(self.salt_bridges_interactions)):
                if self.ligand_profile.loc[
                    self.salt_bridges_interactions.loc[
                        i,'Feat Lig Index'], 'Features_family'] == 'PosIonizable' :
                    x,y,z = (round(self.ligand_profile.loc[
                                 self.salt_bridges_interactions.loc[
                                     i,'Feat Lig Index'], 'x_coord'], 3),
                             round(self.ligand_profile.loc[
                                 self.salt_bridges_interactions.loc[
                                     i,'Feat Lig Index'], 'y_coord'], 3),
                             round(self.ligand_profile.loc[
                                 self.salt_bridges_interactions.loc[
                                     i,'Feat Lig Index'], 'z_coord'], 3))
                    w.write(f'pseudoatom salt_bridges_pos, pos=[{x}, {y}, {z}] \n')
                elif self.pocket_profile.loc[
                    self.salt_bridges_interactions.loc[
                        i,'Feat Pock Index'], 'Features_family'] == 'PosIonizable' :
                    x,y,z = (round(self.pocket_profile.loc[
                                 self.salt_bridges_interactions.loc[
                                     i,'Feat Pock Index'], 'x_coord'], 3),
                             round(self.pocket_profile.loc[
                                 self.salt_bridges_interactions.loc[
                                     i,'Feat Pock Index'], 'y_coord'], 3),
                             round(self.pocket_profile.loc[
                                 self.salt_bridges_interactions.loc[
                                     i,'Feat Pock Index'], 'z_coord'], 3))
                    w.write(f'pseudoatom salt_bridges_pos, pos=[{x}, {y}, {z}] \n')
                if self.ligand_profile.loc[self.salt_bridges_interactions.loc[
                    i,'Feat Lig Index'], 'Features_family'] == 'NegIonizable' :
                    x,y,z = (round(self.ligand_profile.loc[
                                 self.salt_bridges_interactions.loc[
                                     i,'Feat Lig Index'], 'x_coord'], 3),
                             round(self.ligand_profile.loc[
                                 self.salt_bridges_interactions.loc[
                                     i,'Feat Lig Index'], 'y_coord'], 3),
                             round(self.ligand_profile.loc[
                                 self.salt_bridges_interactions.loc[
                                     i,'Feat Lig Index'], 'z_coord'], 3))
                    w.write(f'pseudoatom salt_bridges_neg, pos=[{x}, {y}, {z}] \n')
                elif self.pocket_profile.loc[
                    self.salt_bridges_interactions.loc[
                        i,'Feat Pock Index'], 'Features_family'] == 'NegIonizable' :
                    x,y,z = (round(self.pocket_profile.loc[
                                 self.salt_bridges_interactions.loc[
                                     i,'Feat Pock Index'], 'x_coord'], 3),
                             round(self.pocket_profile.loc[
                                 self.salt_bridges_interactions.loc[
                                     i,'Feat Pock Index'], 'y_coord'], 3),
                             round(self.pocket_profile.loc[
                                 self.salt_bridges_interactions.loc[
                                     i,'Feat Pock Index'], 'z_coord'], 3))
                    w.write(f'pseudoatom salt_bridges_neg, pos=[{x}, {y}, {z}] \n')
            
            w.write('show_as mesh, salt_bridges_pos \n' +
                    'color lightblue, salt_bridges_pos \n' +
                    'show sphere, salt_bridges_pos \n' +
                    'set sphere_color, lightblue, salt_bridges_pos \n' +
                    'set sphere_scale, 2, salt_bridges_pos \n' +
                    'set sphere_transparency,0.7, salt_bridges_pos \n' +
                    'show dots, salt_bridges_pos \n' +
                    'set dot_density, 2, salt_bridges_pos \n' +
                    'set dot_width, 5, salt_bridges_pos \n' +
                    'set dot_solvent, on, salt_bridges_pos \n'
                    'show_as mesh, salt_bridges_neg \n' +
                    'color deepsalmon, salt_bridges_neg \n' +
                    'show sphere, salt_bridges_neg \n' +
                    'set sphere_color, deepsalmon, salt_bridges_neg \n' +
                    'set sphere_scale, 2, salt_bridges_neg \n' +
                    'set sphere_transparency,0.7, salt_bridges_neg \n' +
                    'show dots, salt_bridges_neg \n' +
                    'set dot_density, 2, salt_bridges_neg \n' +
                    'set dot_width, 5, salt_bridges_neg \n' +
                    'set dot_solvent, on, salt_bridges_neg \n'
                    'zoom pocket \n' +
                    'save 3DPharmacophore.pse')
            
        subprocess.run(["pymol", "-cq", "profiler.pml"])
        os.remove('RDKit_ligand.pdb')
        os.remove('complex.pdb')
        os.remove('pocket.pdb')
        os.remove('profiler.pml')
        
        if view == True :
            subprocess.run(["pymol", "3DPharmacophore.pse"])
        if os.path.exists('__pycache__') :
            subprocess.run(["rm", "-rf", "__pycache__"])
    
