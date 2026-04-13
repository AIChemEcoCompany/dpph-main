from rdkit import Chem
import re

element_max_total_bonds = {'H':1, 'B':3, 'C':4, 'N':3, 'O':2, 'Si':4, 'P':5, 'S':6, 'F':1, 'Cl':1, 'Br':1, 'I':1, 
                           'Sn':4, 'Al':3,'Se':2,'As':5,'Sb':5,'Mg':2,'Fr':1,'Te':2}

def valid_smarts(mol:Chem.rdchem.Mol):
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == '*':
            continue
        atom_str = atom.GetSmarts().split('$(')[0]
        # Hs = re.findall('.*[&;]X?H(\d+),?(\d+)?,?(\d+)?', atom_str) #[#6;H1,H2] [#6&H3] max(Hs) ??
        # Hs = re.findall('.*[&;]X?H(\d),?H?(\d)?,?H?(\d)?', atom_str) #[#6;H1,H2] [#6&H3] max(Hs) ?? [#6X4;H2,H3]
        Hs = re.findall('H(\d+)', atom_str) #[#6;H1,H2] [#6&H3] max(Hs) ?? [#6X4;H2,H3]
        Xs = re.findall('X(\d+)', atom_str)#.*&?;?X(\d)
        vs = re.findall('v(\d+)', atom_str)
        if Hs:
            # assert len(Hs) == 1
            try:
                Hs = min([int(i) for i in Hs if i]) #Hs[0]
            except Exception as e:
                print(Hs)
                raise
            atom.SetNumExplicitHs(Hs)
            try:
                Chem.SanitizeMol(mol)
                try:
                    if atom.GetTotalDegree() > element_max_total_bonds[atom.GetSymbol()]:
                        # with open('Hs_log','a') as f:
                        #     f.write(Chem.MolToSmarts(mol) + '\t' + atom.GetSmarts() + '\n')
                        return False
                except:
                    if atom.GetDegree() + Hs > element_max_total_bonds[atom.GetSymbol()]:
                        # with open('Hs_log','a') as f:
                        #     f.write(Chem.MolToSmarts(mol) + '\t' + atom.GetSmarts() + '\n')
                        return False
            except (Chem.rdchem.AtomValenceException, Chem.rdchem.AtomKekulizeException) as e:
                # with open('rdkit AtomValenceException','a') as f:
                #     f.write(Chem.MolToSmarts(mol) + '\t' + atom_str + '\n')
                return False
        if Xs:
            # assert len(Xs) == 1
            Xs = max([int(i) for i in Xs if i])
            if atom.GetDegree()  > int(Xs):
                # with open('degree_log','a') as f:
                #     f.write(Chem.MolToSmarts(mol) + '\t' + atom_str + '\n')
                return False
            if Xs == element_max_total_bonds[atom.GetSymbol()]:
                for bond in atom.GetBonds():
                    if bond.GetBondTypeAsDouble() > 1:
                        return False
        if vs:
            # assert len(vs) == 1
            vs = max([int(i) for i in vs if i])
            if sum(atom.GetBonds()[x].GetBondTypeAsDouble() for x in range(atom.GetDegree()))  > int(vs):
                # with open('valence_log','a') as f:
                #     f.write(Chem.MolToSmarts(mol) + '\t' + atom_str + '\n')
                return False
    return True


if __name__ == "__main__":
    m = Chem.MolFromSmarts("O#C1([C][C][C]1)([C]=N2)[C]3[C]2[C][C][C][C]3")
    res = valid_smarts(m)
    print(res)