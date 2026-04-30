from rdkit import Chem
from rdkit.Chem import AllChem
import re
import json
import pandas as pd
from typing import List, Literal
from collections import defaultdict
from utils.connect_fg import connect_smarts
from pandarallel import pandarallel
from tqdm import tqdm
from rdcanon import canon_smarts
from functools import lru_cache
from utils.get_marked import get_Hatom1, get_inner_ba12

FG_LIST = 'data/priority_fgs.txt'

with open(FG_LIST,'r')as f:
    fg_list = f.read().split('\n')


fg_inner_H = pd.read_csv('data/H_inner_marked.csv',delimiter='\t')

fg_inner_broken_formed = pd.read_csv('data/inner_marked.csv',delimiter='\t')
fg_inner_broken_formed = fg_inner_broken_formed[['smarts', 'smarts_marked']].drop_duplicates()
fg_inner_broken_formed = fg_inner_broken_formed.groupby('smarts')
fg_fg_df = pd.read_csv('data/type4_construct_fg_fg.csv', delimiter='\t')
fg_species = json.load(open('data/fg_species.json'))


@lru_cache(maxsize = 10000)
def get_species(fg:str):
    # script_dir = Path(__file__).parent.parent
    # fg_species = json.load(open('data/fg_species.json','r'))
    for key in fg_species:
        if fg in fg_species[key]:
            return key
    raise KeyError(f'No species {fg}!')
        
def get_smiles_atom_mapping(rxn:str, index, sub_smi:str=None, is_product=False, mol=None)->List[str]:
    rxn = AllChem.ReactionFromSmarts(rxn)
    if not is_product:
        mols = rxn.GetReactants()
    else:
        mols = rxn.GetProducts()
    s2 = Chem.CanonSmiles(Chem.MolToSmiles(Chem.MolFromSmiles(sub_smi), canonical=True, isomericSmiles=False))

    for i, mol in enumerate(mols):#, rxn.GetProducts]:
        mol_ = Chem.Mol(mol)
        for atom in mol_.GetAtoms():
            atom.SetAtomMapNum(0)
        target_mol_smi = Chem.CanonSmiles(Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(mol_, isomericSmiles=False)), isomericSmiles=False))
        #When converting the "rxn" string into a single molecule, the hand shape is lost.  isomericSmiles=False
        if s2 == target_mol_smi and index==i:
            mapNum_idx = {} #mapping -> mol idx 
            for atom in mol.GetAtoms():#
                mapNum_idx[atom.GetAtomMapNum()] = atom.GetIdx()
            return Chem.CanonSmiles(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)), mapNum_idx #('[NH2:2][OH:5]', {2: 0, 5: 1})#
    return "", {}

def deduplicate_by_first_element(data:list):
    seen = set()
    result = []
    
    for item in data:
        key = item[0]
        if key not in seen:
            seen.add(key)
            result.append(item)
    
    return result

def mapping_deal(rxn:str):
    from collections import defaultdict
    
    r, p = rxn.split('>>')
    r, p = r.split('.'), p.split('.')
    r, p = [Chem.MolFromSmiles(x) for x in r], [Chem.MolFromSmiles(x) for x in p]
    
    map_dict = defaultdict(int)
    def cc(rp,map_dict):
        for mol in rp:  
            for atom in mol.GetAtoms():
                map_dict[atom.GetAtomMapNum()] += 1
            for atom in mol.GetAtoms():
                if map_dict[atom.GetAtomMapNum()] != 1:
                    atom.SetAtomMapNum(max(list(map_dict.keys())) + 1)
                map_dict[atom.GetAtomMapNum()] -= 1
    cc(r, defaultdict(int))
    cc(p, defaultdict(int))
    return '.'.join(Chem.MolToSmiles(x) for x in r) + ">>" + '.'.join(Chem.MolToSmiles(x) for x in p)

class Found(Exception):
    pass

class BROKEN_FROMED():
    def __init__(self, rxn_mapping:str, broken_list:List[List[str]], formed_list:List[List[str]], fg_list:List[str] = None):
        self.rxn = rxn_mapping
        self.broken_list = broken_list
        self.formed_list = formed_list
        self.bond_type = {'-':'SINGLE','=':'DOUBLE','#':'TRIPLE',':':"AROMATIC",'aromatic':"AROMATIC"}
        self.shared_data: dict = {}
        self.fg_list = fg_list

        if fg_list is None:
            with open(FG_LIST,'r')as f:
                smarts = f.read().split('\n')
            self.fg_list = smarts
        
        r, p = self.rxn.split(">>")
        self.r, self.p = r.split("."), p.split(".")

        #mol1, mol2, mol3 = Chem.MolFromSmiles(self.r[0]), Chem.MolFromSmiles(self.r[1]), Chem.MolFromSmiles(self.p[0])
        smiles_list = self.r + self.p  
        mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        
        def clean_MapNum(mol):
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            return mol
        mols = [clean_MapNum(mol) for mol in mols]
        # self.r[0], self.r[1], self.p[0] = Chem.MolToSmiles(mol1), Chem.MolToSmiles(mol2), Chem.MolToSmiles(mol3)
        # self.r_mol_list, self.p_mol_list = [(mol1, self.broken_list[0]), (mol2, self.broken_list[1])], [(mol3, self.formed_list[0])]
        self.r_mol_list, self.p_mol_list = [(mol, self.broken_list[i]) for i, mol in enumerate(mols[:len(self.broken_list)])], \
                                            [(mol, self.formed_list[i]) for i, mol in enumerate(list(reversed(mols))[:len(self.formed_list)])]
        
         # A1--A2 + B1--B2 --> A1--B1 + A1--B2 + A2--B1 + A2--B2
        self.inner_broken, self.inner_formed = [], []
        self.r1_broken, self.r1_formed = [], [] # atom1 - atom2
        self.r2_broken, self.r2_formed = [], [] # fg - atom 
        self.r3_broken, self.r3_formed = [], [] # fg - fg
        self.r4_broken, self.r4_formed = [], [] # fg - atom - fg
        self.r5_broken, self.r5_formed = [], [] # fg - atom1 - atom2 - fg
        
    def is_fg_bond_fg_broken(self, mol: Chem.rdchem.Mol, fg:str, bond: str, origin_fg_site: tuple, atom_idx2: int): # fg_bond_fg r2 broken
        for fgfg in self.fg_list:
            fgfg_s = Chem.MolFromSmarts(fgfg)
            fgfg_sites = mol.GetSubstructMatches(fgfg_s)
            for fgfgsite in fgfg_sites:
                if atom_idx2 in fgfgsite and set( origin_fg_site ) & set( fgfgsite ) == set():
                    
                    fg_bond_smarts_sets2 = connect_smarts(fg, fgfg)
                    for fg_bond_smarts_set2 in fg_bond_smarts_sets2:
                        fg_bond_smarts2, fg_bond_smarts_marked, bond2, atom1_symbol2, atom2_symbol2, smarts1, smarts2 = fg_bond_smarts_set2.split("\t")
                        s2_ = Chem.MolFromSmarts(fg_bond_smarts2)
                        fg_bond_site2s = mol.GetSubstructMatches(s2_)
                        for fg_bond_site2 in fg_bond_site2s:
                            if set(fg_bond_site2) == set(origin_fg_site + (atom_idx2,) + fgfgsite) and bond2 == bond:
                                # self.r2_broken.append((index, fg_bond_smarts, broken_, atom1_symbol2, bond2, atom2_symbol2))
                            
                                return fg_bond_smarts2, bond2, atom1_symbol2, atom2_symbol2, smarts1, smarts2
        return [""] * 6 #not found
    def _process(self, 
                 add_atom_smarts: str,
                 fg: str, 
                 mol:Chem.rdchem.Mol,
                 bond_type:Literal['-','=','#','aromatic'], 
                 origin_fg_site: tuple, 
                 add_atom_idx:int, 
                 is_fg_bond_fg_broken = False):
        tmp = Chem.MolFromSmarts(add_atom_smarts)
        if tmp is None:
            add_atom_smarts = "[" + add_atom_smarts + "]"
        fg_bond_smarts_set = connect_smarts(fg, add_atom_smarts) #fg + bond  smarts+f'\t{str(bond)}\t{atom1.GetSymbol()}\t{atom2.GetSymbol()}'
        # self.shared_data['fg'] = fg #
        
        for fg_bond_smarts in fg_bond_smarts_set:
            fg_bond_smarts,fg_bond_smarts_marked, bond, atom1_symbol, atom2_symbol,smarts1,smarts2 = fg_bond_smarts.split("\t") #fg(smarts) - atom(smarts)
            s_ = Chem.MolFromSmarts(fg_bond_smarts)
            fg_bond_sites = mol.GetSubstructMatches(s_)

            for fg_bond_site in fg_bond_sites:
                if set(fg_bond_site) == set( origin_fg_site + (add_atom_idx,)) and self.bond_type[bond_type] == bond:
                    if is_fg_bond_fg_broken: #fg(smarts) - fg(smarts)

                        fg_bond_smarts, bond, atom1_symbol, atom2_symbol, smarts1, smarts2 = self.is_fg_bond_fg_broken(mol, fg, bond, origin_fg_site, add_atom_idx)
                        if not fg_bond_smarts:
                            continue
                    return fg_bond_smarts, atom1_symbol, bond, atom2_symbol, smarts1, smarts2
        
        # print(self.rxn) # not found fg - fg 
        return "", "", "", "","",""
    
    def get_broken_formed(self, mode: str = Literal['r','p'], is_inner_broken = True, is_fg_bond_fg_broken = True):
        if mode == 'r':
            is_product = False
            r_p_mol_list = self.r_mol_list
        elif mode == 'p':
            is_product = True
            r_p_mol_list = self.p_mol_list
        else:
            raise KeyError(mode)
        for index, (mol, brokens) in enumerate(r_p_mol_list):
            if not brokens:
                continue
            try:
                can_smi, mapNum_idx = get_smiles_atom_mapping(self.rxn, index, Chem.MolToSmiles(mol), is_product = is_product)
                if can_smi == "" or mapNum_idx == {}:
                    # print(f'{self.rxn}\n{Chem.MolToSmiles(mol)}')
                    with open('get_broken_formed_error1','a')as f:
                        f.write(f"{self.rxn}\n{self.broken_list}\t{self.formed_list}\n{Chem.MolToSmiles(mol)}")
                    continue
            except TypeError as e:
                continue
                # print(e)
                # print(self.rxn, r_p_mol_list)
                
            for broken_ in brokens: 
                if not isinstance(broken_, str) or broken_ is None:
                    continue    
                if "- H" in broken_: 
                    if not is_inner_broken:
                        continue
                    atom1_symbol2 = re.findall('([A-Z].*):\d+', broken_)[0]
                    atom_ = int(re.findall(':(\d+)', broken_)[0])
                    try:
                        idx_ = mapNum_idx[atom_]
                    except KeyError:
                        break
                    try:
                        # for fg in self.fg_list: ###########################
                        for id, row in fg_inner_H.iterrows(): ###########################adopt broken[:1] [:2] judge [#1:1]-[#12:2]
                            # s = Chem.MolFromSmarts(row['canon_smarts_masked']) 
                            if pd.isna(row['smarts_inner_marked']):
                                continue
                            s = Chem.MolFromSmarts(row['smarts_inner_marked']) #Implicit hydrogen
                            fg_sites = mol.GetSubstructMatches(s) 
                            for fg_site in fg_sites:
                                if idx_ in fg_site:
                                    smarts_atom_idx = list(fg_site).index(idx_)
                                    #for s_index, atom_idx in enumerate(fg_site):
                                        # atom = mol.GetAtomWithIdx(idx_) and atom1_symbol2==s.GetAtomWithIdx(s_index).GetSymbol()
                                    if s.GetAtomWithIdx(smarts_atom_idx).GetAtomMapNum() == 2 : 
                                    # self.r1_broken.append((index, fg, broken_, atom1_symbol2, 'SINGLE', 'H'))
                                        if mode == 'r': #index, fg_bond_smarts, broken_, atom1_symbol, bond, atom2_symbol,smarts1,smarts2
                                            self.inner_broken.append((index, row['smarts_inner_marked'], broken_, atom1_symbol2, 'SINGLE', 'H',row['smarts'],'[H]'))
                                            # self.inner_broken.append((index, row['canon_smarts_masked'], broken_, atom1_symbol2, 'SINGLE', 'H'))
                                        else:
                                            self.inner_formed.append((index, row['smarts_inner_marked'], broken_, atom1_symbol2, 'SINGLE', 'H',row['smarts'],'[H]'))
                                            # self.inner_formed.append((index, row['canon_smarts_masked'], broken_, atom1_symbol2, 'SINGLE', 'H'))
                                
                                        raise Found
                                    # raise AssertionError('not found')
                    except Found:
                        continue
                    continue
                
                try:#fg outer 
                    broken = list(map(int, re.findall(':(\d+)', broken_)))
                    atom1, bond_type, atom2 = re.findall('([A-Z].*?):\d+.*?([-=#]|aromatic).*?([A-Z].*?):\d+', broken_)[0] #|dative
                    atom1_idx, atom2_idx = mapNum_idx[broken[0]], mapNum_idx[broken[1]]

                    assert isinstance(atom1, str)
                    assert isinstance(atom2, str)
                except (IndexError,KeyError) as e:
                    if "dative" not in str(broken_):
                        pass
                        # print(e)
                        # print(mapNum_idx, broken_, broken)
                    continue
                try:
                    if is_inner_broken:
                        try: 
                            for fgs, group in fg_inner_broken_formed:
                                s = Chem.MolFromSmarts(fgs)
                                fg_sites = mol.GetSubstructMatches(s)
                                if not fg_sites:
                                    continue
                                for fg in group['smarts_marked']:#fg_inner_broken_formed['smarts_marked']
                                    if pd.isna(fg):
                                        continue
                                    s = Chem.MolFromSmarts(fg)
                                    fg_sites = mol.GetSubstructMatches(s)

                                    for fg_site in fg_sites:
                                        #Breakage of non-hydrogen bonds within functional groups. Please note that these are the original functional groups.
                                        if atom1_idx in fg_site and atom2_idx in fg_site:#fg inner bf
                                            # self.inner_broken.append([index, fg])#atom1_symbol, bond, atom2_symbol
                                            # Consider the site
                                            mapIdx = {site:i for i, site in enumerate(fg_site)}
                                            s_atom1, s_atom2 = s.GetAtomWithIdx(mapIdx[atom1_idx]), s.GetAtomWithIdx(mapIdx[atom2_idx])
                                            if {s_atom1.GetAtomMapNum(), s_atom2.GetAtomMapNum()} != {1,2}:
                                                continue
                                            #Leave the position information of the disconnected keys
                                            if mode == 'r': #fg inner bf
                                                self.inner_broken.append((index, fg, broken_, atom1, self.bond_type[bond_type], atom2))
                                            else:
                                                self.inner_formed.append((index, fg, broken_, atom1, self.bond_type[bond_type], atom2))
                                            
                                            raise Found
                                            # self.inner_broken.append(index, fg, )
                                     
                        except Found:
                            pass
                    for fg in self.fg_list:
                        s = Chem.MolFromSmarts(fg)
                        fg_sites = mol.GetSubstructMatches(s)

                        for fg_site in fg_sites:
                            try:
                                if (atom1_idx in fg_site) and (atom2_idx not in fg_site):
                                    atom = atom2
                                    atom_idx = atom2_idx
                                                    
                                elif (atom2_idx in fg_site) and (atom1_idx not in fg_site):     
                                    atom = atom1
                                    atom_idx = atom1_idx
                                else:
                                    continue
                                
                                # data = {}
                                # resfg_fg = fg_fg_df[(fg_fg_df['fg1']==fg) & (fg_fg_df['atom2']==atom) & (fg_fg_df['bond']==bond_type)]
                                # if resfg_fg.empty:
                                #     resfg_fg = fg_fg_df[(fg_fg_df['fg2']==fg) & (fg_fg_df['atom1']==atom) & (fg_fg_df['bond']==bond_type)] 
                                # resfg_fg['qmol'] = resfg_fg['fg1_fg2'].appl(Chem.MolFromSmarts)
                                # resfg_fg['fg_index'] = resfg_fg['qmol'].apply(lambda s:mol.GetSubsubsturctMatches(s))
                                # resfg_fg['fg_index'].explode()['fg_index'].apply(lambda idx:set(idx) == set(fg_site))
                                #####################################

                                fg_bond_smarts, atom1_symbol, bond, atom2_symbol,smarts1,smarts2 = self._process(add_atom_smarts = atom, 
                                                                                                fg = fg, 
                                                                                                bond_type = bond_type, 
                                                                                                mol = mol, 
                                                                                                origin_fg_site = fg_site, 
                                                                                                add_atom_idx = atom_idx,
                                                                                                is_fg_bond_fg_broken = is_fg_bond_fg_broken)                        
                                if not fg_bond_smarts:
                                    continue
                                try:
                                    # can_fg_bond_smarts = canon_smarts(fg_bond_smarts)
                                    can_fg_bond_smarts = fg_bond_smarts
                                except:
                                    with open('canon_error','a')as f:
                                        f.write(fg_bond_smarts + '\n')
                                    fg_bond_smarts = fg_fg_df.loc[fg_fg_df['canon_smarts']==can_fg_bond_smarts, 'fg1_fg2']
                                    if len(fg_bond_smarts.value) != 1:
                                        print(fg_bond_smarts,'error')
                                        exit(-1)
                                    fg_bond_smarts = fg_bond_smarts.values[0]
                                if mode == 'r':
                                    self.r2_broken.append((index, can_fg_bond_smarts, broken_, atom1_symbol, bond, atom2_symbol,smarts1,smarts2))
                                else:
                                    self.r2_formed.append((index, can_fg_bond_smarts, broken_, atom1_symbol, bond, atom2_symbol,smarts1,smarts2))
                                
                                # print((mode, index, fg_bond_smarts, broken_, atom1_symbol, bond, atom2_symbol,smarts1,smarts2))
                                raise Found #priority
                            
                            except KeyError as e:
                                print(e,)
                
                except Found:
                    continue
        
        return self.inner_broken,self.inner_formed, self.r2_broken, self.r2_formed


    def get_inner_broken(self):
        self.get_broken_formed('r', is_inner_broken=True)
        self.get_broken_formed('p', is_inner_broken=True)
        res_broken = [x for x in self.inner_broken if x[1]]
        res_formed = [x for x in self.inner_formed if x[1]]
        
        return res_broken, res_formed
    
    def get_fg_fg_broken(self):
        self.get_broken_formed('r', is_inner_broken=True, is_fg_bond_fg_broken=True)
        self.get_broken_formed('p', is_inner_broken=True, is_fg_bond_fg_broken=True)
        res_broken = [x for x in self.r2_broken if x[1]]
        res_formed = [x for x in self.r2_formed if x[1]]
        res_inner_broken = [x for x in self.inner_broken if x[1]]
        res_inner_formed = [x for x in self.inner_formed if x[1]]

        #None data
        res_inner_broken = [[] if x is None else x for x in res_inner_broken]
        res_inner_formed = [[] if x is None else x for x in res_inner_formed]
        res_broken = [[] if x is None else x for x in res_broken]
        res_formed = [[] if x is None else x for x in res_formed]
        return res_inner_broken, res_inner_formed, res_broken, res_formed



def count_fg_freq_classify(df, broken='inner_broken', formed='inner_formed',
                            fg_data_path = 'data/type4_construct_fg_fg.csv',
                            H_data_path='data/H_inner_marked.csv',
                            origin_data_path = 'data/inner_marked.csv',
                            ):

    # df = df[(df[broken]!='[]') | (df[formed]!='[]')]
    res_broken, res_formed = set(), set()
    res_broken_count, res_formed_count = defaultdict(int), defaultdict(int)

    for b in tqdm(df[broken], desc='broken counting...'):
        if isinstance(b,str):
            b = eval(b)
        for item in b:
            res_broken.add(item[1])
            # res_broken_count[f'{item[1]}```{item[-2]}```{item[4]}```{item[-1]}'] +=1
            res_broken_count[item[1]] +=1

    for f in tqdm(df[formed], desc='formed counting...'):
        if isinstance(f,str):
            f = eval(f)
        for item in f:
            res_formed.add(item[1])
            # res_formed_count[f'{item[1]}```{item[-2]}```{item[4]}```{item[-1]}'] += 1
            res_formed_count[item[1]] +=1
    #The inner part is divided into H and non-H.
    if broken == 'inner_broken' and formed == 'inner_formed':
        H_broken, H_formed = defaultdict(int), defaultdict(int)
        for item in list(res_broken_count.keys()):
            if ':1' not in item: #hydrogen
                H_broken[item] +=1
                res_broken_count.pop(item)
        for item in  list(res_formed_count.keys()):
            if ':1' not in item: #hydrogen
                H_formed[item] +=1
                res_formed_count.pop(item)
                
        H_data = pd.read_csv(H_data_path, delimiter='\t')
        mapping = H_data.groupby('smarts_marked_oxygen')['smarts_inner_marked'].first().apply(lambda x: H_broken.get(x, 0)).to_dict() #  Duplicate removal has already been taken into consideration here.
        H_data['broken_freq'] = H_data['smarts_marked_oxygen'].map(mapping)
        mapping = H_data.groupby('smarts_marked_oxygen')['smarts_inner_marked'].first().apply(lambda x: H_formed.get(x, 0)).to_dict() 
        H_data['formed_freq'] = H_data['smarts_marked_oxygen'].map(mapping)
        #Add classification information
        H_data['fg1_species'] = H_data['smarts'].apply(get_species) 
        H_data['fg2_species'] = '*'
        H_data['fg1'] = H_data['smarts']
        H_data['fg2'] = 'H'
        H_data['fg1_fg2'] = H_data['smarts']
        H_data['fg1_fg2_marked'] = H_data['smarts_inner_marked']
        H_data['bond'] = 'SINGLE'
        H_data['atom1'] = H_data['smarts_inner_marked'].apply(get_Hatom1)
        H_data['atom2'] = 'H'
        H_data['fg1_fg2_marked'] = H_data['smarts_inner_marked']
        H_data['canon_smarts'] = H_data['smarts_marked_oxygen']
        H_data = H_data[['fg1', 'fg2', 'fg1_fg2', 'fg1_fg2_marked', 'bond', 'atom1', 'atom2', 'canon_smarts', 
                    'broken_freq', 'formed_freq' ,'fg1_species','fg2_species']]

        H_data.drop_duplicates(subset=['canon_smarts'],inplace=True)
        H_data.to_csv(f"result/{H_data_path.split('.')[0].split('/')[1]}_count.csv", sep='\t', index=False)
        H_data.loc[H_data['broken_freq'].astype(bool)].to_csv(f"result/{H_data_path.split('.')[0].split('/')[1]}_broken_count.csv", sep='\t', index=False)

        #######non-hydrogen
        origin_data = pd.read_csv(origin_data_path, delimiter='\t')
        mapping = origin_data.groupby('smarts_marked_oxygen')['smarts_marked'].first().apply(lambda x: res_broken_count.get(x, 0)).to_dict() 
        origin_data['broken_freq'] = origin_data['smarts_marked_oxygen'].map(mapping)
        mapping = origin_data.groupby('smarts_marked_oxygen')['smarts_marked'].first().apply(lambda x: res_formed_count.get(x, 0)).to_dict()
        origin_data['formed_freq'] = origin_data['smarts_marked_oxygen'].map(mapping)

        #Add classification information
        origin_data['fg1_species'] = origin_data['smarts'].apply(get_species)
        origin_data['fg2_species'] = origin_data['fg1_species'].copy()
        origin_data['fg1'] = origin_data['smarts']
        origin_data['fg2'] = origin_data['smarts']
        origin_data['fg1_fg2'] = origin_data['smarts']
        origin_data['fg1_fg2_marked'] = origin_data['smarts_marked']
        origin_data[['bond','atom1','atom2']] = origin_data.apply(get_inner_ba12,axis=1, result_type='expand')
        origin_data['fg1_fg2_marked'] = origin_data['smarts_marked']
        origin_data['canon_smarts'] = origin_data['smarts_marked_oxygen']
        origin_data = origin_data[['fg1', 'fg2', 'fg1_fg2', 'fg1_fg2_marked', 'bond', 'atom1', 'atom2', 'canon_smarts',
                       'broken_freq', 'formed_freq' ,'fg1_species','fg2_species']]

        origin_data.drop_duplicates(subset=['canon_smarts'],inplace=True)
        origin_data.to_csv(f"result/{origin_data_path.split('.')[0].split('/')[1]}_count.csv", sep='\t', index=False)
        origin_data.loc[origin_data['broken_freq'].astype(bool)].to_csv(f"result/{origin_data_path.split('.')[0].split('/')[1]}_broken_count.csv", sep='\t', index=False)

        # with open(f'result/H_broken_count.json','w')as f:
        #     json.dump(H_broken, f)
        # with open(f'result/H_formed_count.json','w')as f:
        #     json.dump(H_formed, f)        
    
    else:  #fg - fg outer

        fg_data = pd.read_csv(fg_data_path, delimiter='\t')
        fg_data['broken_freq'] = fg_data['fg1_fg2'].apply(lambda x: res_broken_count.get(x, 0)) 
        fg_data['formed_freq'] = fg_data['fg1_fg2'].apply(lambda x: res_formed_count.get(x, 0))
        #Add classification information

        fg_data['fg1_species'] = fg_data['fg1'].apply(get_species) 
        fg_data['fg2_species'] = fg_data['fg2'].apply(get_species) 
        fg_data.drop_duplicates(subset=['canon_smarts'],inplace=True)
        fg_data.to_csv(f"result/{fg_data_path.split('.')[0].split('/')[1]}_count.csv", sep='\t', index=False)
        fg_data[fg_data['broken_freq'].astype(bool)].to_csv(f"result/{fg_data_path.split('.')[0].split('/')[1]}_broken_count.csv", sep='\t', index=False)

    # with open(f'result/{broken}_count.json','w')as f:
    #     json.dump(res_broken_count, f)
    # with open(f'result/{formed}_count.json','w')as f:
    #     json.dump(res_formed_count, f)
    # print(f'{broken.split("_")[0]}_data_count.csv saved in ./result! ')



    
if __name__ == "__main__":
    print('After checking out, the starting calculation broke and formed bonds with the FG!')

    pandarallel.initialize(nb_workers = 16, progress_bar=True)
    def get(row:pd.Series):
        broken_fromed = BROKEN_FROMED(rxn_mapping = row['smiles_am'],
                                        broken_list = row['broken_each_reactant_list'],
                                        formed_list = row['formed_each_product_list'],
                                    fg_list = fg_list
                                    )
        inner_broken, inner_formed, r_broken, r_formed = broken_fromed.get_fg_fg_broken()
        return  inner_broken, inner_formed, r_broken, r_formed
        
    data = pd.read_excel('data/element_bf_checkout.xlsx')
    data['broken_each_reactant_list'] = data['broken_each_reactant_list'].apply(eval)
    data['formed_each_product_list'] = data['formed_each_product_list'].apply(eval) 
    # data['smiles_am'] = data['smiles_am'].apply(mapping_deal)
    data[['inner_broken', 'inner_formed','outer_broken','outer_formed']] = data.parallel_apply(get, axis=1, result_type="expand")
    del data['confidence']

    data.to_csv('result/dpph_bf_result.csv',index=False)
    
    # data = pd.read_csv('dpph23_bf_res.csv')
    
    # # data = pd.read_csv('ab.csv',delimiter='\t')
    # Calculate the frequency of bond breaking at the atomic level, both within and outside functional groups, save the file, and summarize the bond breaking.# 
    count_fg_freq_classify(data,'inner_broken','inner_formed')
    count_fg_freq_classify(data,'outer_broken','outer_formed')

    data = pd.concat([pd.read_csv(f'result/{x}_broken_count.csv',delimiter='\t') for x in ['type4_construct_fg_fg','H_inner_marked','inner_marked']] )
    del data['formed_freq']
    data.to_csv('result/fg_count.csv',index=False)

    print(f'data_count.csv saved in ./result! ')