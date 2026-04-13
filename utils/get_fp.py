from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from scipy.spatial.distance import cdist
from typing import Literal


def get_atoms_with_radius(mol, center_atoms, radius=2):
    res = set(center_atoms)
    for _ in range(radius):
        next_layer = set()
        for idx in res:
            atom = mol.GetAtomWithIdx(idx)
            for neighbor in atom.GetNeighbors():
                next_layer.add(neighbor.GetIdx())
        res.update(next_layer)
    return res

def smol_to_fp(smarts:str, type_=Literal['fg1_fg2','inner','H_inner']):
    '''calc FingerPrint of smarts'''
    s= Chem.MolFromSmarts(smarts)
    # return Chem.PatternFingerprint(s)
    if type_ == 'fg1_fg2' or  type_ == 'inner':
        target_atoms = [a.GetIdx() for a in s.GetAtoms() if a.GetAtomMapNum() in [1, 2]]
        # atoms = set(target_atoms)
        # for idx in target_atoms:
        #     atom = s.GetAtomWithIdx(idx)
        #     for neighbor in atom.GetNeighbors():
        #         atoms.add(neighbor.GetIdx())
        # atoms = [a.GetIdx() for a in s.GetAtoms() if a.GetAtomMapNum() in [1,2]]
        fp = Chem.RDKFingerprint(
            s,
            minPath=1,
            maxPath=3,   
            fpSize=2048,
            fromAtoms=get_atoms_with_radius(s, target_atoms), 
        )
    elif type_ == 'H_inner':
        target_atoms = [a.GetIdx() for a in s.GetAtoms() if a.GetAtomMapNum()==2]
        # atoms = set(target_atoms)
        # for idx in target_atoms:
        #     atom = s.GetAtomWithIdx(idx)
        #     for neighbor in atom.GetNeighbors():
        #         atoms.add(neighbor.GetIdx())
        fp = Chem.RDKFingerprint(
            s,
            minPath=1,
            maxPath=3,   
            fpSize=2048,
            fromAtoms=get_atoms_with_radius(s, target_atoms),
        )
    res = np.array(fp, dtype=bool)
    assert res.sum(), f'invalid fingerprint {smarts}'
    return res

def get_Mfp(s_str:str, type_=Literal['fg1_fg2','inner','H_inner']):
    s = Chem.MolFromSmarts(s_str)
    s.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(s)
    if type_ in ['fg1_fg2','inner']:
        atoms = [a.GetIdx() for a in s.GetAtoms() if a.GetAtomMapNum() in [1,2]]
    elif type_ == 'H_inner':
        atoms = [a.GetIdx() for a in s.GetAtoms() if a.GetAtomMapNum() == 2]
    fp = AllChem.GetMorganFingerprintAsBitVect(s, radius=2, fromAtoms=atoms, nBits=2048)
    res = np.array(fp,dtype=bool)
    assert res.sum(), f'invalid fingerprint {s_str}'
    return res

def tanimoto_np(unk_fp:np.array, known_fps:np.array):
    unk_fp = unk_fp.reshape(1, -1) #.astype(bool)
    known_fps = known_fps#.astype(bool)
    
    # The default distance metric is Jaccard distance. Tanimoto, which is equivalent to Jaccard in the case of binary vectors, is also used.
    distances = cdist(unk_fp, known_fps, metric='jaccard')
    return 1 - distances[0]