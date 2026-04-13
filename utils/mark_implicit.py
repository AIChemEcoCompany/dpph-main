from rdkit import Chem
import re

def marked(smarts:str):
    '''marked implicit Hydrogen as mapping "2" '''
    s = Chem.MolFromSmarts(smarts)
    res_smarts = []
    for atom in s.GetAtoms():
        if atom.GetSymbol() == '*':
            continue
        atom_str = atom.GetSmarts().split('$(')[0]
        Hs = re.findall('H(\d+)', atom_str)
        if Hs:
            atom.SetAtomMapNum(2)
            res_smarts.append(Chem.MolToSmarts(s))
            atom.SetAtomMapNum(0)

    return res_smarts



if __name__ == '__main__':
    s = 'S1[C;H]S[C;!H0][C][C]1'
    res = marked(s)

    print(res)