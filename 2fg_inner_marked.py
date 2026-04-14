from rdkit import Chem
from rdkit.Chem import rdchem
from utils.connect_fg import Found
import pandas as pd
from rdcanon import canon_smarts
import numpy as np
from itertools import combinations_with_replacement
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=8,progress_bar=True)
import re

element_max_total_bonds = {'H':1, 'B':3, 'C':4, 'N':3, 'O':2, 'Si':4, 'P':5, 'S':6, 'F':1, 'Cl':1, 'Br':1, 'I':1,
                            'Sn':4, 'Al':3,'Se':2,'As':5,'Sb':5,'Mg':2,'Zn':2,'Li':1,'Fr':1,'Te':2,'*':1}


def Fr_to_mapping(smarts:str):
    s = Chem.MolFromSmarts(smarts)
    atom_list = []
    try:
        for atom in s.GetAtoms():
            for neighour in atom.GetNeighbors():
                if neighour.GetSymbol() == 'Fr':
                    atom_list.append(atom.GetIdx())
                    if len(atom_list) == 2:
                        raise Found
    except Found:
        pass
    atom_list.sort()
    s.GetAtomWithIdx(atom_list[0]).SetAtomMapNum(1)
    s.GetAtomWithIdx(atom_list[1]).SetAtomMapNum(2)
    
    matches = s.GetSubstructMatches(Chem.MolFromSmarts('[#87]'),)
    indices_to_remove = sorted([match[0] for match in matches], reverse=True) #reverse
    editable_mol = Chem.EditableMol(s)
    
    for idx in indices_to_remove:
        editable_mol.RemoveAtom(idx)
    
    new_mol = editable_mol.GetMol()
    
    
    return canon_smarts(Chem.MolToSmarts(new_mol),mapping=True)

def mark_smarts(smarts: str) -> set:
    smarts_set_, smarts_set = set(), set()

    mol1, mol2, mol3 = Chem.MolFromSmarts(smarts), Chem.MolFromSmarts('[Fr]'), Chem.MolFromSmarts('[Fr]')
    idx1 = range(len(mol1.GetAtoms()))
    
    for atom_idx1 in idx1:
        atom_idx2 = 0
        offset = mol1.GetNumAtoms()
        a1, a2 = atom_idx1, atom_idx2 + offset
        atom1,atom2 = mol1.GetAtomWithIdx(a1), mol2.GetAtomWithIdx(0)
        
        if atom1.GetSymbol() == "*" or atom2.GetSymbol() == "*": # [#6&X4][#12][#9,#17,#35,#53] multi atoms
            break
        
        combined = Chem.CombineMols(mol1, mol2)                                       
        emol = Chem.EditableMol(combined) 
        emol.AddBond(a1, a2, order=rdchem.BondType.SINGLE)  # print(f"Adding bond between {a1} and {a2}")
        new_mol = emol.GetMol()
        # print(Chem.MolToSmarts(new_mol))
        

        ################################
        mol11 = Chem.Mol(new_mol)
        idx2 = range(len(mol11.GetAtoms()))
        for atom_idx11 in idx2:
            atom_idx21 = 0
            offset1 = mol11.GetNumAtoms()
            a11,a21 = atom_idx11, atom_idx21 + offset1
            atom11, atom21 = mol11.GetAtomWithIdx(a11), mol3.GetAtomWithIdx(0)
            if atom11.GetSymbol() == 'Fr':
                continue
            
            for atom_neighbor1 in atom11.GetNeighbors():
                for atom_neighbor2 in atom_neighbor1.GetNeighbors():
                    if atom_neighbor2.GetSymbol() == 'Fr' :
                
                        combined1 = Chem.CombineMols(mol11, mol3)                       
                        emol1 = Chem.EditableMol(combined1) 
                        emol1.AddBond(a11, a21, order=rdchem.BondType.SINGLE)  # print(f"Adding bond between {a1} and {a2}")
                        new_mol1 = emol1.GetMol()        
                        # print(Chem.MolToSmarts(new_mol1))

                        res_smarts = Chem.MolToSmarts(new_mol1)
                        smarts_ = canon_smarts(res_smarts)
                        
        
                        if smarts_ not in smarts_set_:
                            smarts_set_.add(smarts_)# + '\t' + smarts_atom_marked )
                            # new_mol1.GetAtomWithIdx(a1).SetAtomMapNum(1)
                            # new_mol1.GetAtomWithIdx(a2).SetAtomMapNum(2)
                            # smarts_atom_marked = Chem.MolToSmarts(new_mol)
                            
                            # print(smarts_atom_marked)
    for s in smarts_set_:
        s = Fr_to_mapping(s)
        smarts_set.add(s)
    return list(smarts_set)

def mark_smarts_v2(smarts:str):
    s = Chem.MolFromSmarts(smarts)
    res = set()
    for bond in s.GetBonds():
        rwmol = Chem.RWMol(s)
        atom1,atom2 = bond.GetBeginAtom(), bond.GetEndAtom()
        atom1.SetAtomMapNum(1)
        atom2.SetAtomMapNum(2)
        #print(Chem.MolToSmarts(s), atom1.GetIdx(), atom2.GetIdx())
        res_s = Chem.MolToSmarts(s)#,mapping=Truecanon_smarts()
        res.add(res_s)
        atom1.SetAtomMapNum(0)
        atom2.SetAtomMapNum(0)
    return res

def add_bridge_atom_to_each_bond(smarts):
    '''add TRIPLE and oxygen atom'''
    try:
        mol = Chem.MolFromSmarts(smarts)
    except:
        return np.nan
    if mol is None:
        return np.nan
    original_bond_count = mol.GetNumBonds()
    
    for bond_idx in range(original_bond_count):
        rwmol = Chem.RWMol(mol)
        bond = rwmol.GetBondWithIdx(bond_idx)
        atom1_idx = bond.GetBeginAtomIdx()
        atom2_idx = bond.GetEndAtomIdx()
        atom1, atom2 = mol.GetAtomWithIdx(atom1_idx), mol.GetAtomWithIdx(atom2_idx)
        if not (set([atom1.GetAtomMapNum(), atom2.GetAtomMapNum()]) == {1, 2}):
            continue
        
        new_atom_idx = rwmol.AddAtom(Chem.Atom(8))
        
        rwmol.AddBond(atom1_idx, new_atom_idx, Chem.BondType.TRIPLE)
        new_atom_idx = rwmol.AddAtom(Chem.Atom(8))
        rwmol.AddBond(atom2_idx, new_atom_idx, Chem.BondType.TRIPLE)
        m = rwmol.GetMol()
        m.GetAtomWithIdx(atom1_idx).SetAtomMapNum(0)
        m.GetAtomWithIdx(atom2_idx).SetAtomMapNum(0)
        #print(Chem.MolToSmarts(m))
        m_str = canon_smarts(Chem.MolToSmarts(m))
        if Chem.MolFromSmarts(m_str) is not None:
            return m_str
        return Chem.MolToSmarts(m)
    return np.nan

def add_bridge_to_Hatom(smarts):
    '''add TRIPLE and oxygen atom'''
    try:
        mol = Chem.MolFromSmarts(smarts)
    except:
        return np.nan
    if mol is None:
        return np.nan
    
    rwmol = Chem.RWMol(mol)
    new_atom_idx = rwmol.AddAtom(Chem.Atom(8))# add new atom
    global atom
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum()==2:
            # add bond between new atom and atom marked
            rwmol.AddBond(atom.GetIdx(), new_atom_idx, Chem.BondType.TRIPLE)
            break

    m = rwmol.GetMol()
    m.GetAtomWithIdx(atom.GetIdx()).SetAtomMapNum(0)
    
    #print(Chem.MolToSmarts(m))
    return canon_smarts(Chem.MolToSmarts(m))
    
def atom_nums_bonded(atom, atom1_isNH = False)->int:
    """The number of atoms that have bonded"""
    total = 0
    for bond in atom.GetBonds():
        bt = bond.GetBondType()
        if bt == Chem.BondType.SINGLE:
            total += 1
        elif bt == Chem.BondType.DOUBLE:
            total += 2
        elif bt == Chem.BondType.TRIPLE:
            total += 3
        elif bt == Chem.BondType.AROMATIC:
            if atom.GetSymbol() == 'S' and atom.GetIsAromatic():
                return 7
            if atom.GetSymbol() == 'P' and atom.GetIsAromatic():
                return 6
            if atom1_isNH: #[n&H1]
                total-=0.5
            total += 1.5   # 
    total += atom.GetNumExplicitHs()
    
    return total

def marked_inner_H(smarts):
    try:
        mol = Chem.MolFromSmarts(smarts)
    except:
        return np.nan
    if mol is None:
        return np.nan
    ress = []
    mol.UpdatePropertyCache(strict=False)
    # Chem.GetSymmSSSR(mol)
    # Chem.SanitizeMol(mol)
    for atom in mol.GetAtoms():
        atom_str = atom.GetSmarts()
        if atom_str !='[n&H1]':
            Hs = re.findall('H(\d+)', atom_str)
            if not Hs:
                try:
                    # if atom.GetTotalDegree() >= element_max_total_bonds[atom.GetSymbol()] + atom.GetFormalCharge():
                    if atom_str not in ['n', 'p']:
                        if atom_nums_bonded(atom, False) >= element_max_total_bonds[atom.GetSymbol()] + atom.GetFormalCharge():
                            continue
                except:
                    pass
            # elif max(list(map(int, Hs))) == 0:
            elif max(list(map(int, Hs))) == 0:
                continue
        
        atom.SetAtomMapNum(2)
        res = Chem.MolToSmarts(mol)
        ress.append(res)
        atom.SetAtomMapNum(0)
    return ress
    

def test():
    s1 = '[#87]-[#6]([#6]-[#87])([#6])=[#8]'
    s2 = '[#87]-[#6]([#6])([#6])=[#8]-[#87]'
    s3 = '[#6]12[#7;H1][#6]=[#6][#6]1[#7;H1][#6]=[#6]2'

    Fr_to_mapping(s1)
    res = mark_smarts(s2)
    print(res)


if __name__ == '__main__':
    # res = marked_inner_H('[c]12[c]([#7][#6][#6]1)[c][c][c][c]2')
    # res = marked_inner_H('c1ccc2[nH]ccc2c1')  #Kekulization somehow screwed up valence on 4: 0!=1
    # res = marked_inner_H('c12nccc1cccc2')
    # res = marked_inner_H('[N]1[C][C][N][C][C]1')
    # res = marked_inner_H('[#6;H0,H1,H2]([#7;X3H1+0,X3H2+0,X3H0+0])[#6](=[#8])[#8X2H]')
    # res = mark_smarts_v2('[#6][#7+]#[#6-]')
    # print(res)
    # marked_inner_H('[#16v6](=[#8])(=[#8])[#9,#17,#35,#53]')
    
    with open('data/priority_fgs.txt','r')as f:
        smarts = pd.read_csv(f, delimiter='\t', names=['smarts'])

    smarts['smarts_inner_marked'] = smarts['smarts'].parallel_apply(marked_inner_H)
    smarts = smarts.explode('smarts_inner_marked')
    smarts['smarts_marked_oxygen'] = smarts['smarts_inner_marked'].parallel_apply(add_bridge_to_Hatom)
    #smarts['canon_smarts'] = smarts['smarts_inner_marked'].parallel_apply(canon_smarts) 
    print("The number of hydrogen bonds broken within functional groups:", len(smarts.drop_duplicates(subset=['smarts_marked_oxygen'])))

    #save Hdata
    smarts.dropna(subset=['smarts_inner_marked']).to_csv('data/H_inner_marked.csv', index=False, sep='\t')
    
    smarts.drop(columns=['smarts_inner_marked','smarts_marked_oxygen'], inplace=True)
    smarts.drop_duplicates(inplace=True)
    
    ##################################################################################
    with open('data/priority_fgs.txt','r')as f:
        smarts = pd.read_csv(f, delimiter='\t', names=['smarts'])
    #add element bond element in fg-fg
    # from utils.connect_fg import connect_smarts
    # ss = combinations_with_replacement(smarts.loc[smarts['smarts'].str.contains(r'^\[#\d+\]$', regex=True), 'smarts'].values.tolist(), 2)
    # t = pd.DataFrame(ss)
    # for i,row in t.iterrows():
    #     ss = connect_smarts(row[0], row[1])
    #     for s in ss:
    #         smarts.loc[len(smarts), 'smarts'] = s.split('\t')[0]
    smarts['smarts_marked'] = smarts['smarts'].parallel_apply(mark_smarts_v2)
    smarts = smarts.explode('smarts_marked')

    #marked 2 atom with TRIPLE bond and oxygen
    smarts['smarts_marked_oxygen'] = smarts['smarts_marked'].parallel_apply(add_bridge_atom_to_each_bond)
    smarts = smarts.explode('smarts_marked_oxygen')
    
    smarts.drop_duplicates(inplace=True)

    print("The number of bond breakage within functional groups:", len(smarts.drop_duplicates(subset=['smarts_marked_oxygen'])))

    smarts.dropna(subset='smarts_marked').to_csv('data/inner_marked.csv',index=False,sep='\t')

    