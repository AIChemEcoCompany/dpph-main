from rdkit import Chem
import pandas as pd
from rdcanon import canon_smarts

def convert_implicit_H(smarts:str):
    mol = Chem.MolFromSmarts(smarts)
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() == 1:
            raise ValueError(smarts, 'H must be marked 2!')
        if atom.GetAtomMapNum() == 2:
            map_idx = atom.GetIdx()
            break
    rw_mol = Chem.RWMol(mol)
    new_atom_idx = rw_mol.AddAtom(Chem.Atom(8))
    rw_mol.AddBond(map_idx, new_atom_idx, Chem.BondType.SINGLE)
    new_mol = rw_mol.GetMol()
    mm = Chem.MolToSmarts(new_mol)
    
    mm = mm.replace('@','')
    mm = canon_smarts(mm)
    if '!H0' not in mm:
        raise ValueError(smarts, 'add+implicit H error!')
    return mm

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

if __name__ == '__main__':
    res = convert_implicit_H("[#6:2][#7&X3&H0]([#6])[#6])[#6]")
    print(res)