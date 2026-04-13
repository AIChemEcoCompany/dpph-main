from rdkit import Chem
import pandas as pd

def get_Hatom1(smarts:str):
    s = Chem.MolFromSmarts(smarts)
    for atom in s.GetAtoms():
        if atom.GetAtomMapNum() == 2:
            return atom.GetSymbol()

def get_inner_ba12(row:pd.Series):
    smarts = row['smarts_marked']
    s = Chem.MolFromSmarts(smarts)
    for bond in s.GetBonds():
        atom1, atom2 =  bond.GetBeginAtom(), bond.GetEndAtom()
        if atom1.GetAtomMapNum() in [1,2] and atom2.GetAtomMapNum() in [1,2]:
            return bond.GetBondType().__str__(), atom1.GetSymbol(), atom2.GetSymbol()