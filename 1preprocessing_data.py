import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import re
from utils.fracture import get_data, process_row
# from pandarallel import pandarallel
# pandarallel.initialize(nb_workers=32, progress_bar=True)

invalid_char = ["*", "~", "[c]", "[B]", "[O]", "[P]", "[S]", "[N]", "[P]", "[C]", "[I]"]

def get_broken_bond(rxn:str):
    try:
        r_list, p_list = rxn.split('>>')
        r_list, p_list = r_list.split("."), [p_list]
        confidence, mapping_rxn, broken_each_reactant_list, formed_each_product_list, all_bond_changed = process_row(r_list, p_list,balanced = False, consider_broken_inadequacy=True,mapping_tools='RXN')
        return pd.Series([mapping_rxn, confidence, broken_each_reactant_list, formed_each_product_list, all_bond_changed])
    except:
        return pd.Series(["", 0., [], [], []])
    
def valid_rxn(rxn:str, rxn_type:str = 'ac'):
    for char in invalid_char:
        if char in rxn:
            return False
    r, p = rxn.split('>>')
   
    try:
        AllChem.ReactionFromSmarts(rxn)
    except:
        return False
    return True

def clean_rxn(rxn:str):
    try:
        r, p = rxn.split('>>')
        r, p = Chem.MolFromSmiles(r), Chem.MolFromSmiles(p)
        if r is None or p is None:
            return False
        return True
    except:
        return False
def rxn_can(rxn:str):
    try:
        r, p = rxn.split('>>')
        r, p = Chem.CanonSmiles(r), Chem.CanonSmiles(p) 
        if r is None or p is None:
            return rxn
        return r + '>>' + p
    except Exception as e:
        print(e)
        return "False"

def filter_same_sides_regex(text, separator=">>"):
    """Check if the characters on both sides of the delimiter are the same"""
    if not isinstance(text, str):
        return False
    escaped_sep = re.escape(separator)
    pattern = f'^(.+?){escaped_sep}\\1$'
    
    match = re.match(pattern, text)
    return bool(match)

def main():
    data = pd.read_excel('data/DPPH.xlsx')
    print('dealing rxn valid rxn...')
    # df['rxn'] = data['A']
    print('dealing rxn canioncial rxn...')
    data['valid'] = data['rxn'].apply(clean_rxn)
    data = data[data['valid']]
    data['rxn'] = data['rxn'].apply(rxn_can)
    
    data['str_count'] = data['rxn'].apply(filter_same_sides_regex)
    data = data[~data['str_count']]
    data.drop_duplicates(inplace=True)
    data['str_count'] = data["rxn"].apply(lambda x: len(x))

    # data = data[data['str_count']<=512]
    del data['str_count']
    del data['valid']

    print('The remaining data volume：', len(data))
    data[[ 'smiles_am', 'confidence', 'broken_each_reactant_list', 'formed_each_product_list', 'all_bond_changed']] = data['rxn'].apply(get_broken_bond)
    #data = data[data['confidence'] >= CONFIDENCE]

    for col in ['broken_each_reactant_list', 'formed_each_product_list', 'all_bond_changed']:
        data[col] = data[col].apply(lambda x:str(x).replace('None','[]'))
    
    data.drop_duplicates(subset='rxn',inplace=True)
    data.to_csv('dpph_rxn_mapping.csv', sep='\t', index=None) 

    print('Atomic tracking may not be entirely accurate. Some reactions require manual inspection!')
    print('preprocessing done!')

if __name__ == '__main__':
    main()
    
    # rxn = 'COc1ccc(Br)cc1OC>>COc1ccc(C2CCCC(CC(=O)OC(C)C)O2)cc1OC.COc1ccc([C@@H]2CCC[C@H](CC(=O)OC(C)C)O2)cc1OC'
    # valid_rxn(rxn)

    # data = pd.read_csv('test.csv', delimiter='\t')#.sample(1000) #############################sample#########################################
    # data.columns = ['rxn']
    # data["valid"] = data['rxn'].apply(lambda x : valid_rxn(x, 'ac'))
    # data['str_count'] = data["rxn"].apply(lambda x: len(x))

    # data = data[(data['valid'] == True) & (data['str_count']<=512)]
    
    # del data['str_count']
    # del data['valid']

    
    # data[[ 'smiles_am', 'confidence', 'broken_each_reactant_list', 'formed_each_product_list', 'all_bond_changed']] = data['rxn'].parallel_apply(get_broken_bond)
    # data = data[data['confidence'] >= 0.2]

    # for col in ['broken_each_reactant_list', 'formed_each_product_list', 'all_bond_changed']:
    #     data[col] = data[col].parallel_apply(lambda x:str(x).replace('None','[]'))
    
    # data[['rxn_type']] = 'ac'
    
    # data.to_csv('test2.csv',sep='\t',index=False)