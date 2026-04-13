import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from loguru import logger
from typing import Literal
try:
    import sys
    sys.path.append("..") 
    from modules.atom_mapping import data_process
except:
    from modules.atom_mapping import data_process 
from chem_balancer.main import masterbalance
from modules.rxn4bond import RXN


from tqdm import tqdm

def get_atom_mapping(r_smiles_list, p_smiles_list,rxn_SMILES=None, mapper:str=Literal['RXN','indigo','local'], balanced = False):
    if balanced:
        r_mw = 0
        p_mw = 0
        for r_smiles in r_smiles_list:
            try:
                r_mol = Chem.MolFromSmiles(r_smiles)
                r_mw += Descriptors.MolWt(r_mol)
            except Exception as e:
                logger.error(e)
                logger.error(r_smiles)
                print("\n --Failed to obtain the molecular weight of the substrate")
                return False, False, [], [], [], 0.   #confidence
            
        for p_smiles in p_smiles_list:
            try:
                p_mol = Chem.MolFromSmiles(p_smiles)
                p_mw += Descriptors.MolWt(p_mol)
            except Exception as e:
                logger.error(e)
                logger.error(p_smiles)
                print("\n --Failed to obtain the molecular weight of the substrate")
                return False, False, [], [], [], 0.

        to_be_balanced = False
        if len(r_smiles_list) == 0 or len(p_smiles_list) == 0:
            return False, False, [], [], [], 0.
        elif r_mw < p_mw:
            # Attempt to match and construct reaction smiles
            to_be_balanced = True
            rxn_SMILES = ".".join(r_smiles_list) + ">>" + ".".join(p_smiles_list)
            try:
                finaldf = masterbalance(rxn_SMILES,ncpus=1)
                balanced_rxn_SMILES = finaldf["balrxnsmiles"].values[0]
                if len(balanced_rxn_SMILES.split(">>")) == 2:
                    r_smiles_list = balanced_rxn_SMILES.split(">>")[0].split(".")
                    p_smiles_list = balanced_rxn_SMILES.split(">>")[1].split(".")
                else:
                    print("\n --After balancing, there is no substrate or product left")
                    return to_be_balanced, False, [], [], [], 0.
            except Exception as e:
                print("\n --Balancing failed")
                return to_be_balanced, False, [], [], [], 0.
    else:
        to_be_balanced = True
    r_list = []

    for smiles in r_smiles_list:
        r_list.append({
            'role': 'substrate',
            "smiles": smiles
        })
    for smiles in p_smiles_list:
        r_list.append({
            'role': 'product',
            "smiles": smiles
        })

    reaction_list = [r_list]
    # start = time.time()
    
    try:
        # if mapper == 'localmapper':
        #     mapper = localmapper()
        
        result_am, mapping_confidence = data_process(reaction_list, mapper)
    except:
        return to_be_balanced, False, [], [],  0.

    if len(result_am) == 0:
        return to_be_balanced, False, [], [], 0.
    else:
        result_am = result_am[0]
        sub_list = []
        prod_list = []
        for item in result_am:
            if "role" in item.keys():
                if "mapped_smiles" not in item.keys():
                    return to_be_balanced, False, [], [], 0.
                else:
                    if item["role"] == "substrate":
                        sub_list.append(item["mapped_smiles"])
                    if item["role"] == "product":
                        prod_list.append(item["mapped_smiles"])
    return to_be_balanced, True, sub_list, prod_list, mapping_confidence

def extract_used_numbers(molecules):
        used_numbers = set()
        for mol in molecules:
            for atom in mol.GetAtoms():
                map_num = atom.GetAtomMapNum()
                if map_num != 0:
                    used_numbers.add(map_num)
        return used_numbers

def assign_new_numbers(molecules, used_numbers):
        max_used_number = max(used_numbers, default=0)
        for mol in molecules:
            for atom in mol.GetAtoms():
                if atom.GetAtomMapNum() == 0:
                    max_used_number += 1
                    atom.SetAtomMapNum(max_used_number)
                    used_numbers.add(max_used_number)
        return molecules, used_numbers

def get_bond(am_sub_list, am_prod_list, consider_broken_inadequacy = False,changed_is_bf=True):
        r_list = am_sub_list
        p_list = am_prod_list
        if len(r_list) == 0 or len(p_list) == 0:
            return False, [], [], [], [], [], None

        try:
            rxn = RXN(r_list, p_list, am=False)
        except Exception as e:
            logger.error("Error occurred while creating the rxn object: {}".format(e))
            return False, [], [], [], [], [], None

        try:
            broken, formed, bond_changed = rxn.get_bond_change(with_idx=True, consider_broken_inadequacy = consider_broken_inadequacy)
            broken_each_reactant, formed_each_product = rxn.get_bond_change_for_each()
            
            if changed_is_bf:
                for c in bond_changed:
                    broken.append(c.split('->')[0].strip())
                    formed.append(c.split('->')[1].strip())
                    
            # broken, formed, bond_changed remove duplicates separately
            all_broken = list(set(broken))
            all_formed = list(set(formed))
            all_bond_changed = list(set(bond_changed))
            
            r_list

            # The breaking of bonds for each substrate
            broken_each_reactant_list = []
            for idx, item in enumerate(broken_each_reactant):
                if len(item) > 0:
                    item = list(set(item))
                    broken_each_reactant_list.append(item)
                else:
                    broken_each_reactant_list.append(None)

            # The bonding of each product
            formed_each_product_list = []
            for idx, item in enumerate(formed_each_product):
                if len(item) > 0:
                    item = list(set(item))
                    formed_each_product_list.append(item)
                else:
                    formed_each_product_list.append(None)
            
            return True, all_broken, all_formed, all_bond_changed, broken_each_reactant_list, formed_each_product_list, rxn
        except Exception as e:
            logger.error("Error occurred while get_bond_change: {}".format(e))
            return False, [], [], [], [], [], None

def process_row(r_smiles:list,p_smiles:list,mapping_tools='RXN', balanced = False, consider_broken_inadequacy=False,mapped=True):
        # mapping_tools rxn_mapping indigo 
        # self.condition_id = self.row[self.condition_id_col_name]
        
        to_be_balanced, am_status, am_sub_list, am_prod_list, confidence = True, True,r_smiles,p_smiles,1.0
        # atom mapping 
        if mapped:
            to_be_balanced, am_status, am_sub_list, am_prod_list, confidence = get_atom_mapping(r_smiles,p_smiles, balanced = balanced,mapper=mapping_tools)

        # Convert SMILES to RDKit Molecule objects
        am_sub_mols = [Chem.MolFromSmiles(smiles) for smiles in am_sub_list]
        am_prod_mols = [Chem.MolFromSmiles(smiles) for smiles in am_prod_list]

        # For am_sub_list and self.am_prod_list, check whether all atoms in the SMILES have an atom mapping. If not, add atom mapping numbers throughout the reaction.        used_numbers = extract_used_numbers(am_sub_mols + am_prod_mols)
        am_sub_mols, used_numbers = assign_new_numbers(am_sub_mols, used_numbers)
        am_prod_mols, used_numbers = assign_new_numbers(am_prod_mols, used_numbers)

        # Convert RDKit Molecule objects back to SMILES
        new_am_sub_list = [Chem.MolToSmiles(mol, True) for mol in am_sub_mols]
        new_am_prod_list = [Chem.MolToSmiles(mol, True) for mol in am_prod_mols]

        am_sub_list = new_am_sub_list
        am_prod_list = new_am_prod_list

        if am_status:
            # Consider weak disconnections
            bond_status, \
            all_broken, \
            all_formed, \
            all_bond_changed, \
            broken_each_reactant_list, \
            formed_each_product_list, \
            rxn_obj \
            = get_bond(am_sub_list, am_prod_list, consider_broken_inadequacy=consider_broken_inadequacy) 
            smiles_am = rxn_obj.rxn_smiles
        else:
            bond_status = False
            print("\n[get atom mapping error]")
            print(r_smiles,p_smiles)
            return 0.0, "", [],[]
            # return ''.join(r_smiles)+">>"+''.join(p_smiles)
        # return '\t'.join(map(str,[smiles_am, confidence, bond_status, all_broken, all_formed, all_bond_changed, \
        #         broken_each_reactant_list, formed_each_product_list,])) 
        return   confidence, smiles_am, broken_each_reactant_list, formed_each_product_list, all_bond_changed




def get_data(row:str):
    try:
        r, p = row['rxn'].split(">>")
        confidence, smiles_am, broken_each_reactant_list,formed_each_product_list  = process_row([r], [p], balanced = False)

        return pd.Series([confidence, smiles_am, broken_each_reactant_list,formed_each_product_list], index = ['confidence', 'smiles_am', 'broken_each_reactant_list','formed_each_product_list'])
    except:
        return pd.Series([0.,"",[],[]],index= ['confidence', 'smiles_am', 'broken_each_reactant_list','formed_each_product_list'])
    


if __name__ == '__main__':

    rxn = "[CH3:1][CH2:2]/[C:3]([N:4]1[CH2:5][CH2:6][CH2:7][CH2:8]1)=[CH:9]\[CH3:10].[O:21]=[C:20]([O:22][CH3:23])[CH:19]2[C:12]([c:13]3[cH:18][cH:17][cH:16][cH:15][cH:14]3)=[N:11]2>>[CH3:1][CH2:2][C:3]4([NH:11][C:19]([C:20]([O:22][CH3:23])=[O:21])=[C:12]([CH:9]4[CH3:10])[c:13]5[cH:18][cH:17][cH:16][cH:15][cH:14]5)[N:4]6[CH2:5][CH2:6][CH2:7][CH2:8]6"
    rxn = "C#CCOc1ccc2c(C)cc(=O)oc2c1.CCOC(=O)CN=[N+]=[N-]>>CCOC(=O)Cn1cc(COc2ccc3c(C)cc(=O)oc3c2)nn1"
    rxn = '[NH2:11]OP(c1ccccc1)(c2ccccc2)=O.[NH2:1]OP(c3ccccc3)(c4ccccc4)=O.[F:5][c:4]5[c:6](=[O:7])[nH:8][c:9]([nH:2][cH:3]5)=[O:10].[Na]>>[NH2:1][n:2]([cH:3][c:4]([F:5])[c:6](=[O:7])[n:8]6[NH2:11])[c:9]6=[O:10]'
    r, p = rxn.split(">>")
    # r_list = [x.split(".") for x in r_list]
    r_list = r.split(".")
    p_list = p.split(".")
    def get(rxn:str):
        r, p = rxn.split(">>")
        r_list,p_list = r.split("."), p.split(".")
        confidence, smiles_am, broken_each_reactant_list, formed_each_product_list, all_bond_changed = process_row(r_list, p_list, mapped=False)
        return pd.Series([confidence, smiles_am,broken_each_reactant_list, formed_each_product_list,all_bond_changed])
    
    #After checkout, recalculate the bond breaking and bonding of atoms.
    df = pd.read_excel('/data/DPPH断成键3-18.xlsx',sheet_name='去重') #Manually modify the atomic mapping information
    from pandarallel import pandarallel
    pandarallel.initialize(nb_workers=16,progress_bar=True)
    # df[['broken_each_reactant_list', 'formed_each_product_list','all_bond_changed']] = df['smiles_am'].parallel_apply(get)
    df[['confidence','smiles_am','broken_each_reactant_list', 'formed_each_product_list','all_bond_changed']] = df['smiles_am'].apply(get)
    # res = process_row(r_list, p_list,mapping=False)
    # print(res)
    df.to_excel('../element_bf.xlsx')
    print(df.loc[11])