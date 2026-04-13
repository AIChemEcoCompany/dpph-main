from rdkit import Chem
from rdkit.Chem import rdchem
from rdcanon import canon_smarts
from rdkit import RDLogger
import pandas as pd
from itertools import product
from functools import lru_cache
import copy
import os
import re
import pandas as pd
from tqdm import tqdm
import json
from multiprocessing import Pool
import itertools
from pathlib import Path
from func_timeout import func_timeout

RDLogger.DisableLog('rdApp.*')
script_dir = Path(__file__).parent.parent
with open(script_dir / 'data/exclude_santi.txt')as f:
    exclude_santi = f.read().split('\n')
    
bonds = [rdchem.BondType.SINGLE,
         #rdchem.BondType.AROMATIC,
         rdchem.BondType.DOUBLE,
         rdchem.BondType.TRIPLE]



element_max_total_bonds = {'H':1, 'B':3, 'C':4, 'N':3, 'O':2, 'Si':4, 'P':5, 'S':6, 'F':1, 'Cl':1, 'Br':1, 'I':1, 
                           'Sn':4, 'Al':3,'Se':2,'As':5,'Sb':5,'Mg':2,'Fr':1,'Te':2} #'Zn':2,'Li':1,

bonds_int = {'SINGLE':1,'AROMATIC':1.5,'DOUBLE':2,'TRIPLE':3,'UNSPECIFIED':1}

class Found(Exception):
    pass

@lru_cache
def get_ele_nums_bonds(ele1, ele2, )->int:
    script_dir = Path(__file__).parent.parent
    ele_bonds = pd.read_csv(script_dir / 'data/ele/ele.csv', encoding='utf-8', index_col=0)
    if ele_bonds[ele1][ele2]:
        return int(ele_bonds[ele1][ele2])
    else:
        return int(ele_bonds[ele2][ele1])
    
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
            total += 1.5   
    total += atom.GetNumExplicitHs()
    bonds= [bond.GetBondTypeAsDouble() for bond in atom.GetBonds()]
    return total#sum(bonds)

exclude = ['Cu', 'K', 'Pd', 'Ni', 'Ti', 'Sr', 'W', 'Nd', 'Rh', 'Eu', 'Sm', 'At', 'Ce', 'Ir',
            'Pr', 'Ar', 'Ac', 'Cs', 'Dy', 'Ta', 'Gd', 'Re',
            'Ho', 'Sc', 'Ru', 'V', 'Cr', 'Ge', 'Cd', 'Rb','Ag',
            'Be', 'Th', 'Ga', 'La', 'Bi', 'U', 'Y',
            'Tb', 'Tc', 'In', 'Lu', 'Os', 'Xe', 'Nb', 'Mo', 'Tl'
            ]

aromatic_atoms = ['C','N','O','S','P']


def valid_smarts(mol:Chem.rdchem.Mol,idxa,idxb):
    for atom in mol.GetAtoms():
        if atom.GetIdx() in [idxa,idxb]:
            continue
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
            mol_ = copy.deepcopy(mol)
            try:
                # Chem.SanitizeMol(mol_)
                try:
                    if atom.GetTotalDegree() > element_max_total_bonds[atom.GetSymbol()] + atom.GetFormalCharge(): #Only check the constraints rather than the specific chemical environment
                        # with open('Hs_log','a') as f:
                        #     f.write(Chem.MolToSmarts(mol) + '\t' + atom.GetSmarts() + '\n')
                        return False
                except:
                    if atom.GetDegree() + Hs > element_max_total_bonds[atom.GetSymbol()] + atom.GetFormalCharge():
                        # with open('Hs_log','a') as f:
                        #     f.write(Chem.MolToSmarts(mol) + '\t' + atom.GetSmarts() + '\n')
                        return False
            except (Chem.rdchem.AtomValenceException) as e: #, Chem.rdchem.AtomKekulizeException
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
                    if bond.GetBondTypeAsDouble() > 1 and atom.GetFormalCharge()==0:
                        return False
        if vs:
            # assert len(vs) == 1
            vs = max([int(i) for i in vs if i])
            if sum(atom.GetBonds()[x].GetBondTypeAsDouble() for x in range(atom.GetDegree()))  > int(vs):
                # with open('valence_log','a') as f:
                #     f.write(Chem.MolToSmarts(mol) + '\t' + atom_str + '\n')
                return False
    return True



def connect_smarts(smarts1: str, smarts2: str, Add_Fr:bool = False, connect_Fr:bool = False, broken_atom_marked:bool = True) -> set:
    """
        Add_Fr // add a [Fr] atom in atom2(smarts2)
        connect_Fr // connect both smarts containing Fr atom
    """
    smiles_set, smarts_set = set(), set()

    mol1, mol2, mol3= Chem.MolFromSmarts(smarts1), Chem.MolFromSmarts(smarts2), Chem.MolFromSmarts('[Fr]')
    if mol1 is None or mol2 is None:
        
        with open("error",'a')as f:
            f.write(f'{smarts1} {smarts2}\n')
        return set()

    if connect_Fr:
        Fr_idx1 = [atom.GetIdx() for atom in mol1.GetAtoms() if atom.GetSymbol() == 'Fr'][0]
        Fr_idx2 = [atom.GetIdx() for atom in mol2.GetAtoms() if atom.GetSymbol() == 'Fr'][0]
        end_atom_idx1, end_atom_idx2 = None, None
        try:
            for atom in mol1.GetAtoms(): 
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetSymbol() == 'Fr':
                        end_atom_idx1 = atom.GetIdx()
                        raise Found
        except Found:
            pass
        try:
            for atom in mol2.GetAtoms(): 
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetSymbol() == 'Fr':
                        end_atom_idx2 = atom.GetIdx()
                        raise Found
        except Found:
            pass
        if end_atom_idx1 is None or end_atom_idx2 is None:
            print(smarts1,smarts2)
        assert end_atom_idx1 is not None
        assert end_atom_idx2 is not None
        #delete Fr atom
        editable_mol1, editable_mol2 = Chem.EditableMol(mol1),Chem.EditableMol(mol2)
        editable_mol1.RemoveAtom(Fr_idx1), editable_mol2.RemoveAtom(Fr_idx2)
        mol1, mol2 = editable_mol1.GetMol(), editable_mol2.GetMol()

    idx1,idx2 = range(len(mol1.GetAtoms())), range(len(mol2.GetAtoms()))
    
    for atom_idx1 in idx1:
        for atom_idx2 in idx2:
            
            if connect_Fr:
                if atom_idx1 != end_atom_idx1 or atom_idx2 != end_atom_idx2:
                    continue
            # Obtain the global index of the connected atom (note the offset of the atomic index after merging)
            offset = mol1.GetNumAtoms()
            a1,a2 = atom_idx1, atom_idx2 + offset
            atom1,atom2 = mol1.GetAtomWithIdx(a1), mol2.GetAtomWithIdx(a2 - offset)
            atom1_isNH,atom2_isNH = False, False
            if atom1.GetIsAromatic() and atom1.GetSmarts() == '[n&H1]':
                atom1_isNH = True
            if atom2.GetIsAromatic() and atom2.GetSmarts() == '[n&H1]':
                atom2_isNH = True
            
            for bond in bonds:
                # Creating editable molecular objects requires reconfiguration
                combined = Chem.CombineMols(copy.deepcopy(mol1), copy.deepcopy(mol2))
                
                if atom1.GetSymbol() in exclude or atom2.GetSymbol() in exclude : ######
                    break
                
                if atom1.GetIsAromatic() or atom2.GetIsAromatic():
                    if bond in [rdchem.BondType.DOUBLE, rdchem.BondType.TRIPLE]:
                        break
                   
                emol = Chem.EditableMol(combined) 
                for atom__ in [a1, a2]:
                    if combined.GetAtomWithIdx(atom__).GetSmarts() == '[n&H1]':
                        new_atom_n = Chem.AtomFromSmarts(f"[n]")
                        emol.ReplaceAtom(atom__, new_atom_n)

                emol.AddBond(a1, a2, order=bond)  # print(f"Adding bond between {a1} and {a2}")
                new_mol = emol.GetMol()
                # try:
                #     Chem.SanitizeMol(new_mol)
                # except Exception as e: #(Chem.rdchem.AtomValenceException,Chem.rdchem.AtomKekulizeException)
                #      continue

                #maximum bonding, bond order, determination
                if atom1.GetSymbol() == "*" or atom2.GetSymbol() == "*": # [#6&X4][#12][#9,#17,#35,#53] multi atoms
                    break
                
                if new_mol.GetBondBetweenAtoms(a1, a2).GetBondType().__str__() == "DATIVE":
                    continue
                # if atom1.GetSymbol() in exclude_element or atom1.GetSymbol() in exclude_element:
                #     continue
                #Maximum bonding number determination
                bond_type = new_mol.GetBondBetweenAtoms(a1, a2).GetBondType().__str__() #consider UNSPECIFIED
                if bonds_int[bond_type] > get_ele_nums_bonds(atom1.GetSymbol(), atom2.GetSymbol()):
                    break
                        
                #The current number of bonds is greater than the maximum number of bonds. The aromatic nitrogen requires special attention. [nH]
                a1_atom,a2_atom = new_mol.GetAtomWithIdx(a1),new_mol.GetAtomWithIdx(a2)
                # if atom_nums_bonded(a1_atom) == 4 and atom_nums_bonded(a2_atom):

                if atom_nums_bonded(a1_atom,atom1_isNH) > element_max_total_bonds[a1_atom.GetSymbol()] + a1_atom.GetFormalCharge():#atom1
                    break
                if atom_nums_bonded(a2_atom,atom2_isNH) > element_max_total_bonds[a2_atom.GetSymbol()] + a2_atom.GetFormalCharge(): #atom2
                    break
                new_mol_ = copy.deepcopy(new_mol)
                
                if smarts1 not in exclude_santi and smarts2 not in exclude_santi:
                    try:
                        # new_mol_.UpdatePropertyCache(strict=False)
                        # Chem.SanitizeMol(new_mol_)
                        pass
                    except Exception as e: #(Chem.rdchem.AtomValenceException,Chem.rdchem.AtomKekulizeException)
                        print(Chem.MolToSmarts(new_mol))
                        continue
                
                    #Mark the positions with [Fr]
                    if Add_Fr:
                        new_mol2 = copy.deepcopy(new_mol)
                        combined2= Chem.CombineMols(new_mol2, mol3)
                        emol2 = Chem.EditableMol(combined2)
                        try:
                            emol2.AddBond(a2, a2+1, order=rdchem.BondType.SINGLE)
                        except:
                            break
                    # print(f"Adding bond between {a1} and {a2}")
                        new_mol2 = emol2.GetMol() 
                        try:
                            Chem.SanitizeMol(new_mol2)
                        except (Chem.rdchem.AtomValenceException,Chem.rdchem.AtomKekulizeException) as e:
                            break
                        
                        atom2 = new_mol2.GetAtomWithIdx(a2)
                        if atom_nums_bonded(atom2) > element_max_total_bonds[atom2.GetSymbol()]:
                            break    
                        new_mol = new_mol2

                    #Check whether the smarts are reasonable like c1(ccccc1)-[#6&H3][#6]=[#8]
                    if not valid_smarts(new_mol,a1,a2):
                        break

                smiles = Chem.MolToSmiles(new_mol) #[#6][#6]-[#9,#17,#35,#53] [$([CX4][OH1])&!$(C([OH1])[#7,#8,#15,#16,#9,#17,#35,#53])]
                smarts = Chem.MolToSmarts(new_mol)
                
                if Chem.MolFromSmiles(smiles) is None:
                    m_ = Chem.MolFromSmiles(smiles, sanitize=False)
                    # if atom1.GetIsAromatic() and atom2.GetIsAromatic():
                    try:
                        Chem.SanitizeMol(m_, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    except Chem.rdchem.AtomValenceException as e: #ignore kekulized
                        break
                # if '<-' in smarts or '->' in smarts or '.' in smarts:
                #         continue
                # smarts = canon_smarts(smarts)

                # broken_atom_marked = True    
                if smiles not in smiles_set:
                    smarts_atom_marked = ""
                    if broken_atom_marked:
                        # broken_atom_marked as 1 and 2
                        mol1_, mol2_ = Chem.Mol(mol1), Chem.Mol(mol2)
                        mol1_.GetAtomWithIdx(atom_idx1).SetAtomMapNum(1)
                        mol2_.GetAtomWithIdx(atom_idx2).SetAtomMapNum(2)
                        fg1_marked = Chem.MolToSmarts(mol1_)
                        fg2_marked = Chem.MolToSmarts(mol2_)

                        new_mol.GetAtomWithIdx(a1).SetAtomMapNum(1)
                        new_mol.GetAtomWithIdx(a2).SetAtomMapNum(2)
                        smarts_atom_marked = Chem.MolToSmarts(new_mol)
                    # if Add_Fr:
                    #     smiles_set.add(smiles)
                    #     smarts_set.add(smarts + '\t' + smarts_atom_marked + f'\t{str(bond)}\t{atom1.GetSymbol()}\t{atom2.GetSymbol()}\t{smarts1}\t{smarts2}')
                    #     continue
                    smiles_set.add(smiles)
                    smarts_set.add(smarts + '\t' + smarts_atom_marked + f'\t{str(bond)}\t{atom1.GetSymbol()}\t{atom2.GetSymbol()}\t{smarts1}\t{smarts2}')
                    # smarts_set.add(smarts + '\t' + smarts_atom_marked + f'\t{str(bond)}\t{atom1.GetSymbol()}\t{atom2.GetSymbol()}\t{smarts1}\t{smarts2}\t{fg1_marked}\t{fg2_marked}')
                    # print(smarts)
                    
                    # smarts_set.add(smarts)
                    # except Exception as e:
                        # with open('error_broken','a')as f:
                        #     f.write(f'{smarts1} {smarts2} \nerror {e}\n')
                   
      
   
    return smarts_set

def process_pair(args,):
    """Parallel functions for processing individual molecules in combination"""
    fg1, fg2 = args
    tmp = connect_smarts(fg1, fg2, connect_Fr=False)  #
    return [(fg1, fg2, i) for i in tmp]

def process_pair31(args,):
    """Parallel functions for processing individual molecules in combination"""
    fg1, fg2 = args
    tmp = connect_smarts(fg1, fg2, Add_Fr = True, connect_Fr=False)  
    return [(fg1, fg2, i) for i in tmp]

def process_pair32(args,):
    """Parallel functions for processing individual molecules in combination"""
    fg1, fg2 = args
    tmp = connect_smarts(fg1, fg2, Add_Fr = False, connect_Fr=True)  
    return [(fg1, fg2, i) for i in tmp]

def merge_fg1_fg2(file = "fg1fg2_kekulize_test.txt"):
    with open(r"fgs4_30.txt",'r')as f:
        fgs1 = f.read().split('\n')
    with open(r"fgs4_30_ultra.txt",'r')as f:
        fgs2 = f.read().split('\n')
    fgs = fgs1 + fgs2
    ATOM = ['[H]', 'B', 'C', 'N', 'O', 'F', '[Al]', '[Si]', 'P', 'S', 'Cl', 'Br', '[Sn]', 'I'][1:][::-1] 

    # pairs = list(itertools.product(fgs, repeat=2))
    # pairs = list(itertools.combinations_with_replacement(fgs,2))
    pairs = list(itertools.product(fgs,ATOM))

    # print(len(pairs))

    with Pool(processes=64) as pool:  
        results = []
        with tqdm(total=len(pairs), desc="Processing pairs") as pbar:
            for result in pool.imap_unordered(process_pair, pairs):
                results.extend(result)
                pbar.update()

    df = pd.DataFrame(results, columns=["fg1", "fg2", "fg1_fg2"])
    df[['fg1_fg2', 'bond','atom1','atom2']] = df['fg1_fg2'].str.split('\t', expand=True)
    df.drop_duplicates(subset=['fg1_fg2'], keep='first',inplace=True)
    # df.to_csv("fg1fg2_kekulize4_30.txt", sep='\t', index=False,)
    df.to_csv(file, sep='\t', index=False,)




def can_(smarts, timeout=1000):
    try:
        result = func_timeout(timeout, canon_smarts,(smarts,True)) #reserved map
        return result
    except:
        with open('canon_error','a')as f:
            f.write(smarts + '\n')
        return smarts
     
def remove_87(smarts:str):
    m = Chem.MolFromSmarts(smarts)
    for atom in m.GetAtoms():
        if atom.GetSymbol() == 'Fr':
            emol = Chem.EditableMol(m)
            emol.RemoveAtom(atom.GetIdx())
            return Chem.MolToSmarts(emol.GetMol())
        
@lru_cache(maxsize = 10000)
def get_species(fg:str):
    script_dir = Path(__file__).parent.parent
    fg_species = json.load(open(script_dir / 'data/fg_species.json','r'))
    for key in fg_species:
        if fg in fg_species[key]:
            return key

def main4():
    from pandarallel import pandarallel
    pandarallel.initialize(nb_workers=32,progress_bar=True)

    res_file_path = 'data/type4_construct_fg_fg.csv'
    script_dir = Path(__file__).parent.parent
    #if not os.path.exists(script_dir / res_file_path):
    with open('data/priority_fgs.txt','r')as f:
        fgs0 = f.read().split('\n')
    pairs = list(itertools.combinations_with_replacement(fgs0, 2))#[:10]
    with Pool(processes = 32) as pool:  
        results = []
        with tqdm(total=len(pairs), desc="Processing pairs") as pbar:
            for result in pool.imap_unordered(process_pair, pairs):
                results.extend(result)
                pbar.update()

    df = pd.DataFrame(results, columns=["fg1", "fg2", "fg1_fg2"]) # 'smarts1_marked','smarts2_marked'
    
    # df[['fg1_fg2','fg1_fg2_marked', 'bond','atom1','atom2','smarts1','smarts2','smarts1_marked','smarts2_marked']] = df['fg1_fg2'].str.split('\t', expand=True)
    df[['fg1_fg2','fg1_fg2_marked', 'bond','atom1','atom2','smarts1','smarts2']] = df['fg1_fg2'].str.split('\t', expand=True)
    
   
    ##### remove dup ######
    original_index = df.index 
    df_shuffled = df.sample(frac=1)
    df_shuffled['canon_smarts'] = df_shuffled['fg1_fg2'].parallel_apply(can_)
    df = df_shuffled.reindex(original_index)
    
    try:
        del df['smarts1']
        del df['smarts2']
    except:
        pass
    df.drop_duplicates(subset=['canon_smarts'], keep='first',inplace=True)
    #####remove dup ######
    
    #classify
    # df['fg1_category'] = df['fg1'].apply(get_species)  
    # df['fg2_category'] = df['fg2'].apply(get_species)
    
    df.to_csv(res_file_path, sep='\t', index=False)
    # df.to_parquet('data/type4_construct_fg_fg.parquet', index=False)

def construct_fg_main12(construct_type:int = 1):
    '''main1 or main2'''
    script_dir = Path(__file__).parent.parent
    with open(script_dir / 'data/priority_fgs.txt','r')as f:
        fgs0 = f.read().split('\n')
    atoms = ['[#1]','[#5]','[#6]','[#7]','[#8]','[#9]','[#12]','[#13]','[#14]','[#15]','[#16]','[#17]','[#33]','[#34]','[#35]','[#50]','[#52]','[#53]' ]
        #B C N O F Mg Al Si P S Cl As Se Br Sn I
    if construct_type == 1: #[atom1]-[atom2]
        pairs = list(itertools.combinations_with_replacement(atoms, 2)) #include self-self
        res_file_path = f'type{construct_type}_construct_atom_atom.csv'
    elif construct_type == 2: #[fg]-[atom2]
        pairs = list(itertools.product(atoms, fgs0))
        res_file_path = f'type{construct_type}_construct_fg_atom.csv'
    # elif construct_type == 3: #[fg1]-[atom]-[fg2]
        # res_file_path = f'type{construct_type}_construct_fg_atom_fg.csv'

    
    # elif construct_type == 4: #[fg1]-[fg2]
    #     res_file_path = f'type{construct_type}_construct_fg_fg.csv'
    #     main()
    #     return
    # elif construct_type == 5:
    #     res_file_path = f'type{construct_type}_construct_fg_atom_atom_fg.csv'
    else:
        raise RuntimeError("The parameter 'construct_type' must be one of [1,2,3,4]!")
    
    from pandarallel import pandarallel
    pandarallel.initialize(nb_workers=8,progress_bar=True)

    # if not os.path.exists(script_dir / res_file_path):
    with Pool(processes = 4) as pool:  
        results = []
        with tqdm(total=len(pairs), desc=f"Processing {res_file_path.split('.')[-1]} pairs") as pbar:
            for result in pool.imap_unordered(process_pair, pairs):
                results.extend(result)
                pbar.update()

    df = pd.DataFrame(results, columns=["fg1", "fg2", "fg1_fg2"])
    df[['fg1_fg2','fg1_fg2_marked', 'bond','atom1','atom2','smarts1','smarts2']] = df['fg1_fg2'].str.split('\t', expand=True)
    
    # else:    
    #     df = pd.read_csv(script_dir / res_file_path, delimiter='\t')

    original_index = df.index 
    df_shuffled = df.sample(frac=1)
    df_shuffled['canon_smarts'] = df_shuffled['fg1_fg2'].parallel_apply(can_)
    df = df_shuffled.reindex(original_index)
    df['canon_smarts_masked'] = df_shuffled['fg1_fg2_marked'].parallel_apply(can_)
    try:
        del df['smarts1']
        del df['smarts2']
    except:
        pass
    df.drop_duplicates(subset=['canon_smarts'], keep='first',inplace=True)
    df.to_csv(script_dir / res_file_path, sep='\t', index=False,)
    print(f'saved in {script_dir / res_file_path}.')


def main3():
    script_dir = Path(__file__).parent.parent
    with open(script_dir / 'data/priority_fgs.txt','r')as f:
        fgs0 = f.read().split('\n')
    atoms = ['[#1]','[#5]','[#6]','[#7]','[#8]','[#9]','[#12]','[#13]','[#14]','[#15]','[#16]','[#17]','[#33]','[#34]','[#35]','[#50]','[#53]' ]
        #B C N O F Mg Al Si P S Cl As Se Br Sn I
    
    res_file_path = f'type{3}_construct_fg_atom_fg.csv'
    
    pairs1 = list(itertools.product(fgs0, atoms)) #a2 add Fr
    
    from pandarallel import pandarallel
    pandarallel.initialize(nb_workers=4, progress_bar=True)

    # if not os.path.exists(script_dir / res_file_path):
    with Pool(processes = 4) as pool:  
        results = []
        with tqdm(total=len(pairs1), desc=f"Processing {res_file_path.split('.')[-1]} pairs") as pbar:
            for result in pool.imap_unordered(process_pair31, pairs1):
                results.extend(result)
                pbar.update()

    df1 = pd.DataFrame(results, columns=["fg1", "fg2", "fg1_fg2"])
    df1[['fg1_fg2','fg1_fg2_marked', 'bond','atom1','atom2','smarts1','smarts2']] = df1['fg1_fg2'].str.split('\t', expand=True)
    df1.to_csv('type3_construct_fg_atom_Fr.csv', sep='\t', index=False)
    # else:    
    #     df = pd.read_csv(script_dir / res_file_path, delimiter='\t')
    #connect Fr
    # pairs = list(itertools.combinations_with_replacement(df.fg1_fg2,2))
    #fg-atom-Fr  fg-Fr

    pairs2 = list(itertools.product(fgs0, ['[#87]']))
    with Pool(processes = 4) as pool:  
        results = []
        with tqdm(total=len(pairs2), desc=f"Processing {res_file_path.split('.')[-1]} pairs") as pbar:
            for result in pool.imap_unordered(process_pair, pairs2):
                results.extend(result)
                pbar.update()

    df2 = pd.DataFrame(results, columns=["fg1", "fg2", "fg1_fg2"])
    df2[['fg1_fg2','fg1_fg2_marked', 'bond','atom1','atom2','smarts1','smarts2']] = df2['fg1_fg2'].str.split('\t', expand=True)
    df1.to_csv('type3_construct_fg_Fr.csv', sep='\t', index=False)

    pairs3 = list(itertools.product(df1.fg1_fg2, df2.fg1_fg2))#[:1000]
    with Pool(processes = 32) as pool:  
        results = []
        with tqdm(total=len(pairs3), desc=f"Processing {res_file_path.split('.')[-1]} pairs") as pbar:
            for result in pool.imap_unordered(process_pair32, pairs3):
                results.extend(result)
                pbar.update()
        print(len(results)) 

   

if __name__ == '__main__':
    # from pandarallel import pandarallel
    # pandarallel.initialize(nb_workers=32)

    # atoms = ['[#9]','[#17]','[#35]','[#53]', '[#33]', '[#5]', '[#6]', '[#12]', '[#7]', '[#8]', '[#15]', '[#16]', '[#51]', '[#34]', '[#14]', '[#50]']
    # pairs = list(itertools.combinations_with_replacement(atoms, 2))
    # res = connect_smarts('[#14]', '[#50]')
    # for x,y in pairs:
    #     res = connect_smarts(x,y)
    #construct_fg(2)
    # res = connect_smarts('c12[nH]ccc1cccc2','c12[nH]ccc1cccc2')
    #res = connect_smarts('c12[nH]ccc1cccc2','[#7]')
    # res = connect_smarts('[c]1([s][n][c]2(=[#8]))[c]2[c][c][c][c]1','[#7]')
    res = connect_smarts('[c]1[c][c][c][c][c]1','[c]1[c][c][c][c][c]1[#7;!a;X4H+,X4H2+,X4H3+,X3H+0,X3H0+0,X3H2+0;!$([#7][!#6])]')
    print(res)
    print(len(res))
    
    